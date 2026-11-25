// Fast Matrix-Free Geometric Multigrid FEM Solver for TMS Induced Electric Field
// Reference: I. Laakso and A. Hirata, Phys. Med. Biol. 57 (2012) 7753-7765
//
// Features:
//   - Zero-dependency NIfTI-1 (.nii) reader/writer for SimNIBS charm segmentations
//     (optional .nii.gz support when compiled with -DUSE_ZLIB -lz)
//   - SimNIBS .ccd magnetic dipole coil parser with 4x4 matsimnibs transform
//   - OpenMP multithreading (8-color checkerboard SOR, compatible with MSVC/GCC/Clang)
//   - Anti-staircasing 3x3x3 conductivity smoothing
//
// Build (Linux/macOS/MinGW-w64):
//   g++ -std=c++17 -O3 -march=native -fopenmp main.cpp -o tms_gmg_solver
// Build (with zlib for direct .nii.gz I/O):
//   g++ -std=c++17 -O3 -march=native -fopenmp -DUSE_ZLIB main.cpp -lz -o tms_gmg_solver
// Build (Windows MSVC Developer Command Prompt):
//   cl /std:c++17 /O2 /openmp /EHsc main.cpp /Fe:tms_gmg_solver.exe
//
// Usage:
//   ./tms_gmg_solver <final_tissues_MNI.nii> <coil.ccd> <out_efield.nii> [options]
// Options:
//   --matsimnibs <file>   4x4 coil transform matrix file (mm in 4th column)
//   --didt <val>          Coil current rate dI/dt in A/s (default: 1e6)
//   --levels <L>          Number of multigrid levels (default: 4)
//   --tol <val>           Relative residual tolerance (default: 1e-5)
//   --max-iter <N>        Maximum V-cycle iterations (default: 30)
//   --no-smooth           Disable 3x3x3 conductivity smoothing

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_ZLIB
#include <zlib.h>
#endif

// Portable OpenMP macros (supports OpenMP 2.0 in MSVC as well as OpenMP 4.0+ in GCC/Clang)
#if defined(_OPENMP) && (_OPENMP >= 201307)
    #define OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for collapse(2) schedule(static)")
    #define OMP_SIMD_RED_AXAYAZ        _Pragma("omp simd reduction(+:ax, ay, az)")
#elif defined(_OPENMP)
    #define OMP_PARALLEL_FOR_COLLAPSE2 _Pragma("omp parallel for schedule(static)")
    #define OMP_SIMD_RED_AXAYAZ
#else
    #define OMP_PARALLEL_FOR_COLLAPSE2
    #define OMP_SIMD_RED_AXAYAZ
#endif

/*
    1. NIfTI-1 Data Structures & I/O
*/
#pragma pack(push, 1)
struct Nifti1Header {
    int32_t sizeof_hdr;
    char    data_type[10];
    char    db_name[18];
    int32_t extents;
    int16_t session_error;
    char    regular;
    char    dim_info;
    int16_t dim[8];
    float   intent_p1, intent_p2, intent_p3;
    int16_t intent_code;
    int16_t datatype;
    int16_t bitpix;
    int16_t slice_start;
    float   pixdim[8];
    float   vox_offset;
    float   scl_slope;
    float   scl_inter;
    int16_t slice_end;
    char    slice_code;
    char    xyzt_units;
    float   cal_max, cal_min;
    float   slice_duration;
    float   toffset;
    int32_t glmax, glmin;
    char    descrip[80];
    char    aux_file[24];
    int16_t qform_code, sform_code;
    float   quatern_b, quatern_c, quatern_d;
    float   qoffset_x, qoffset_y, qoffset_z;
    float   srow_x[4], srow_y[4], srow_z[4];
    char    intent_name[16];
    char    magic[4];
};
#pragma pack(pop)
static_assert(sizeof(Nifti1Header) == 348, "NIfTI-1 header must be 348 bytes");

struct NiftiVolume {
    Nifti1Header hdr{};
    int nx = 0, ny = 0, nz = 0;
    double dx_m = 0.0, dy_m = 0.0, dz_m = 0.0; // Voxel spacing in meters
    float srow_x[4]{}, srow_y[4]{}, srow_z[4]{}; // Voxel-to-world affine (mm)
    std::vector<int> labels;
};

static void build_srow_from_qform(const Nifti1Header& hdr,
                                  float sx[4], float sy[4], float sz[4]) {
    double b = hdr.quatern_b;
    double c = hdr.quatern_c;
    double d = hdr.quatern_d;
    double a2 = 1.0 - (b * b + c * c + d * d);
    double a = (a2 > 0.0) ? std::sqrt(a2) : 0.0;

    double r11 = a * a + b * b - c * c - d * d;
    double r12 = 2.0 * (b * c - a * d);
    double r13 = 2.0 * (b * d + a * c);
    double r21 = 2.0 * (b * c + a * d);
    double r22 = a * a + c * c - b * b - d * d;
    double r23 = 2.0 * (c * d - a * b);
    double r31 = 2.0 * (b * d - a * c);
    double r32 = 2.0 * (c * d + a * b);
    double r33 = a * a + d * d - b * b - c * c;

    double qfac = (hdr.pixdim[0] < 0.0f) ? -1.0 : 1.0;
    double dx = std::abs(hdr.pixdim[1]);
    double dy = std::abs(hdr.pixdim[2]);
    double dz = std::abs(hdr.pixdim[3]) * qfac;

    sx[0] = static_cast<float>(r11 * dx); sx[1] = static_cast<float>(r12 * dy);
    sx[2] = static_cast<float>(r13 * dz); sx[3] = hdr.qoffset_x;

    sy[0] = static_cast<float>(r21 * dx); sy[1] = static_cast<float>(r22 * dy);
    sy[2] = static_cast<float>(r23 * dz); sy[3] = hdr.qoffset_y;

    sz[0] = static_cast<float>(r31 * dx); sz[1] = static_cast<float>(r32 * dy);
    sz[2] = static_cast<float>(r33 * dz); sz[3] = hdr.qoffset_z;
}

static bool has_gzip_magic(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    unsigned char magic[2] = {0, 0};
    ifs.read(reinterpret_cast<char*>(magic), 2);
    return ifs && magic[0] == 0x1f && magic[1] == 0x8b;
}

NiftiVolume load_charm_nifti(const std::string& path) {
    Nifti1Header hdr{};
    std::vector<uint8_t> payload;

#ifdef USE_ZLIB
    gzFile fp = gzopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Failed to open NIfTI file: " + path);
    if (gzread(fp, &hdr, sizeof(Nifti1Header)) != sizeof(Nifti1Header) || hdr.sizeof_hdr != 348) {
        gzclose(fp);
        throw std::runtime_error("Invalid or byte-swapped NIfTI-1 header: " + path);
    }
    const size_t n_vox = static_cast<size_t>(hdr.dim[1]) * hdr.dim[2] * hdr.dim[3];
    const size_t bytes_per_vox = static_cast<size_t>(hdr.bitpix / 8);
    const size_t total_bytes = n_vox * bytes_per_vox;
    if (gzseek(fp, static_cast<z_off_t>(hdr.vox_offset), SEEK_SET) < 0) {
        gzclose(fp);
        throw std::runtime_error("Failed to seek to NIfTI voxel data.");
    }
    payload.resize(total_bytes);
    size_t offset = 0;
    while (offset < total_bytes) {
        unsigned int chunk = static_cast<unsigned int>(std::min<size_t>(total_bytes - offset, 1u << 28));
        int got = gzread(fp, payload.data() + offset, chunk);
        if (got <= 0) {
            gzclose(fp);
            throw std::runtime_error("Truncated NIfTI voxel payload.");
        }
        offset += static_cast<size_t>(got);
    }
    gzclose(fp);
#else
    if (has_gzip_magic(path)) {
        throw std::runtime_error(
            "File '" + path + "' is gzip-compressed (.nii.gz), but this binary was built without zlib.\n"
            "Either decompress it first (e.g., 'gzip -dk " + path + "' or 7-Zip) "
            "or recompile with -DUSE_ZLIB -lz.");
    }
    std::ifstream fp(path, std::ios::binary);
    if (!fp) throw std::runtime_error("Failed to open NIfTI file: " + path);
    fp.read(reinterpret_cast<char*>(&hdr), sizeof(Nifti1Header));
    if (!fp || hdr.sizeof_hdr != 348) {
        throw std::runtime_error("Invalid or byte-swapped NIfTI-1 header: " + path);
    }
    const size_t n_vox = static_cast<size_t>(hdr.dim[1]) * hdr.dim[2] * hdr.dim[3];
    const size_t bytes_per_vox = static_cast<size_t>(hdr.bitpix / 8);
    const size_t total_bytes = n_vox * bytes_per_vox;
    fp.seekg(static_cast<std::streamoff>(hdr.vox_offset), std::ios::beg);
    payload.resize(total_bytes);
    fp.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(total_bytes));
    if (!fp) throw std::runtime_error("Truncated NIfTI voxel payload.");
#endif

    NiftiVolume vol;
    vol.hdr = hdr;
    vol.nx = hdr.dim[1];
    vol.ny = hdr.dim[2];
    vol.nz = hdr.dim[3];
    vol.dx_m = std::abs(hdr.pixdim[1]) * 1e-3;
    vol.dy_m = std::abs(hdr.pixdim[2]) * 1e-3;
    vol.dz_m = std::abs(hdr.pixdim[3]) * 1e-3;

    if (hdr.sform_code > 0) {
        std::memcpy(vol.srow_x, hdr.srow_x, sizeof(vol.srow_x));
        std::memcpy(vol.srow_y, hdr.srow_y, sizeof(vol.srow_y));
        std::memcpy(vol.srow_z, hdr.srow_z, sizeof(vol.srow_z));
    } else if (hdr.qform_code > 0) {
        build_srow_from_qform(hdr, vol.srow_x, vol.srow_y, vol.srow_z);
    } else {
        vol.srow_x[0] = std::abs(hdr.pixdim[1]);
        vol.srow_y[1] = std::abs(hdr.pixdim[2]);
        vol.srow_z[2] = std::abs(hdr.pixdim[3]);
    }

    const size_t n_vox_check = static_cast<size_t>(vol.nx) * vol.ny * vol.nz;
    vol.labels.resize(n_vox_check);
    const float slope = (hdr.scl_slope == 0.0f) ? 1.0f : hdr.scl_slope;
    const float inter = hdr.scl_inter;

    auto decode = [&](auto dummy) {
        using T = decltype(dummy);
        const T* raw = reinterpret_cast<const T*>(payload.data());
        for (size_t i = 0; i < n_vox_check; ++i) {
            vol.labels[i] = static_cast<int>(std::lround(static_cast<double>(raw[i]) * slope + inter));
        }
    };

    switch (hdr.datatype) {
        case 2:   decode(uint8_t{});  break; // DT_UINT8 (standard charm output)
        case 4:   decode(int16_t{});  break; // DT_INT16
        case 8:   decode(int32_t{});  break; // DT_INT32
        case 512: decode(uint16_t{}); break; // DT_UINT16
        case 16:  decode(float{});    break; // DT_FLOAT32
        case 64:  decode(double{});   break; // DT_FLOAT64
        default:
            throw std::runtime_error("Unsupported NIfTI datatype code: " + std::to_string(hdr.datatype));
    }

    return vol;
}

void save_nifti_float32(const std::string& out_path,
                        const Nifti1Header& ref_hdr,
                        const std::vector<float>& data,
                        int nt = 1) {
    Nifti1Header hdr = ref_hdr;
    hdr.dim[0] = (nt > 1) ? 4 : 3;
    hdr.dim[4] = static_cast<int16_t>(nt);
    hdr.dim[5] = 1;
    hdr.dim[6] = 1;
    hdr.dim[7] = 1;
    hdr.pixdim[4] = (nt > 1) ? 1.0f : 0.0f;
    hdr.intent_code = 0;
    hdr.datatype = 16; // DT_FLOAT32
    hdr.bitpix = 32;
    hdr.vox_offset = 352.0f;
    hdr.scl_slope = 1.0f;
    hdr.scl_inter = 0.0f;
    if (data.empty()) {
        hdr.cal_min = 0.0f;
        hdr.cal_max = 0.0f;
    } else {
        auto [mn, mx] = std::minmax_element(data.begin(), data.end());
        hdr.cal_min = *mn;
        hdr.cal_max = *mx;
    }
    std::memcpy(hdr.magic, "n+1\0", 4);

    const uint8_t extender[4] = {0, 0, 0, 0};

#ifdef USE_ZLIB
    const bool write_gz = (out_path.size() >= 3 && out_path.substr(out_path.size() - 3) == ".gz");
    if (write_gz) {
        gzFile fp = gzopen(out_path.c_str(), "wb6");
        if (!fp) throw std::runtime_error("Failed to create NIfTI file: " + out_path);
        gzwrite(fp, &hdr, sizeof(Nifti1Header));
        gzwrite(fp, extender, sizeof(extender));
        const uint8_t* ptr = reinterpret_cast<const uint8_t*>(data.data());
        const size_t total_bytes = data.size() * sizeof(float);
        size_t offset = 0;
        while (offset < total_bytes) {
            unsigned int chunk = static_cast<unsigned int>(std::min<size_t>(total_bytes - offset, 1u << 28));
            if (gzwrite(fp, ptr + offset, chunk) != static_cast<int>(chunk)) {
                gzclose(fp);
                throw std::runtime_error("Failed writing gzip NIfTI payload.");
            }
            offset += chunk;
        }
        gzclose(fp);
        return;
    }
#endif

    std::ofstream fp(out_path, std::ios::binary);
    if (!fp) throw std::runtime_error("Failed to create NIfTI file: " + out_path);
    fp.write(reinterpret_cast<const char*>(&hdr), sizeof(Nifti1Header));
    fp.write(reinterpret_cast<const char*>(extender), sizeof(extender));
    fp.write(reinterpret_cast<const char*>(data.data()),
             static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!fp) throw std::runtime_error("Failed writing NIfTI payload.");
}

/*
    2. Multigrid Grid Hierarchy & Conductivity Mapping
*/
struct GridLevel {
    int nx, ny, nz;            // Number of voxels
    double h;                  // Voxel side length (meters)
    std::vector<double> sigma; // Voxel conductivities: nx * ny * nz
    std::vector<double> phi;   // Nodal scalar potential: (nx+1)*(ny+1)*(nz+1)
    std::vector<double> rhs;   // Nodal source term f:   (nx+1)*(ny+1)*(nz+1)
    std::vector<double> res;   // Nodal residual r:      (nx+1)*(ny+1)*(nz+1)

    GridLevel(int vx, int vy, int vz, double spacing)
        : nx(vx), ny(vy), nz(vz), h(spacing),
          sigma(static_cast<size_t>(vx) * vy * vz, 0.0),
          phi(static_cast<size_t>(vx + 1) * (vy + 1) * (vz + 1), 0.0),
          rhs(static_cast<size_t>(vx + 1) * (vy + 1) * (vz + 1), 0.0),
          res(static_cast<size_t>(vx + 1) * (vy + 1) * (vz + 1), 0.0) {}

    inline size_t v_idx(int i, int j, int k) const {
        return (static_cast<size_t>(k) * ny + j) * nx + i;
    }

    inline size_t n_idx(int i, int j, int k) const {
        return (static_cast<size_t>(k) * (ny + 1) + j) * (nx + 1) + i;
    }
};

struct PadInfo {
    int pad_x = 0, pad_y = 0, pad_z = 0;
};

// SimNIBS charm integer label -> isotropic conductivity (S/m)
double charm_label_to_sigma(int label) {
    switch (label) {
        case 1:  return 0.126; // White Matter (WM)
        case 2:  return 0.275; // Gray Matter (GM)
        case 3:  return 1.654; // Cerebrospinal Fluid (CSF)
        case 4:  return 0.010; // Bone (general)
        case 5:  return 0.465; // Scalp / Skin
        case 6:  return 0.500; // Eye balls / Vitreous humor
        case 7:  return 0.008; // Compact bone
        case 8:  return 0.025; // Spongy bone
        case 9:  return 0.600; // Blood
        case 10: return 0.160; // Muscle
        default: return 0.0;   // 0 = Air / background
    }
}

std::vector<GridLevel> init_hierarchy_from_nifti(const NiftiVolume& vol,
                                                 int num_levels,
                                                 PadInfo& pad_out) {
    if (num_levels < 2) throw std::runtime_error("num_levels must be >= 2");

    const int mult = 1 << (num_levels - 1);
    auto pad_dim = [mult](int n) {
        int min_padded = n + 4; // >= 2 air voxels per side (safe for 3x3x3 smoothing + boundary)
        int rem = min_padded % mult;
        return (rem == 0) ? min_padded : (min_padded + mult - rem);
    };

    const int nx0 = pad_dim(vol.nx);
    const int ny0 = pad_dim(vol.ny);
    const int nz0 = pad_dim(vol.nz);
    pad_out = {(nx0 - vol.nx) / 2, (ny0 - vol.ny) / 2, (nz0 - vol.nz) / 2};

    std::vector<GridLevel> levels;
    levels.reserve(num_levels);
    for (int l = 0; l < num_levels; ++l) {
        int scale = 1 << l;
        levels.emplace_back(nx0 / scale, ny0 / scale, nz0 / scale, vol.dx_m * scale);
    }

    GridLevel& fine = levels[0];
    OMP_PARALLEL_FOR_COLLAPSE2
    for (int k = 0; k < vol.nz; ++k) {
        for (int j = 0; j < vol.ny; ++j) {
            for (int i = 0; i < vol.nx; ++i) {
                size_t src = (static_cast<size_t>(k) * vol.ny + j) * vol.nx + i;
                size_t dst = fine.v_idx(i + pad_out.pad_x, j + pad_out.pad_y, k + pad_out.pad_z);
                fine.sigma[dst] = charm_label_to_sigma(vol.labels[src]);
            }
        }
    }

    return levels;
}

void smooth_conductivity_3x3x3(GridLevel& g) {
    std::vector<double> smoothed(g.sigma.size(), 0.0);
    constexpr double inv27 = 1.0 / 27.0;

    OMP_PARALLEL_FOR_COLLAPSE2
    for (int k = 1; k < g.nz - 1; ++k) {
        for (int j = 1; j < g.ny - 1; ++j) {
            for (int i = 1; i < g.nx - 1; ++i) {
                double sum = 0.0;
                for (int dz = -1; dz <= 1; ++dz)
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                            sum += g.sigma[g.v_idx(i + dx, j + dy, k + dz)];
                smoothed[g.v_idx(i, j, k)] = sum * inv27;
            }
        }
    }
    g.sigma.swap(smoothed);
}

void build_coarse_conductivities(std::vector<GridLevel>& levels) {
    for (size_t l = 0; l + 1 < levels.size(); ++l) {
        const GridLevel& fine = levels[l];
        GridLevel& coarse = levels[l + 1];

        OMP_PARALLEL_FOR_COLLAPSE2
        for (int k = 0; k < coarse.nz; ++k) {
            for (int j = 0; j < coarse.ny; ++j) {
                for (int i = 0; i < coarse.nx; ++i) {
                    double sum = 0.0;
                    for (int dz = 0; dz < 2; ++dz)
                        for (int dy = 0; dy < 2; ++dy)
                            for (int dx = 0; dx < 2; ++dx)
                                sum += fine.sigma[fine.v_idx(2 * i + dx, 2 * j + dy, 2 * k + dz)];
                    coarse.sigma[coarse.v_idx(i, j, k)] = 0.125 * sum;
                }
            }
        }
    }
}

/*
    3. SimNIBS .ccd Coil Parser & Vector Potential A0'
*/
struct CoilDipoles {
    std::vector<float> px, py, pz; // World positions (meters)
    std::vector<float> mx, my, mz; // World magnetic moments (m^2 per A)
    size_t size() const { return px.size(); }
};

using Mat4 = std::array<std::array<double, 4>, 4>;

Mat4 load_matsimnibs(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Failed to open matsimnibs file: " + path);
    Mat4 m{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (!(ifs >> m[r][c])) {
                throw std::runtime_error("Expected 16 numeric entries in matsimnibs file: " + path);
            }
        }
    }
    return m;
}

CoilDipoles load_ccd_file(const std::string& ccd_path, const Mat4& matsimnibs) {
    std::ifstream ifs(ccd_path);
    if (!ifs) throw std::runtime_error("Failed to open .ccd file: " + ccd_path);

    CoilDipoles dipoles;
    std::string line;
    while (std::getline(ifs, line)) {
        auto first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos || line[first] == '#') continue;

        std::istringstream iss(line);
        double x, y, z, mx, my, mz;
        if (!(iss >> x >> y >> z >> mx >> my >> mz)) continue; // Skip header count line

        // Transform position (coil m -> world m; matsimnibs translation is in mm)
        double wx = matsimnibs[0][0] * x + matsimnibs[0][1] * y + matsimnibs[0][2] * z + matsimnibs[0][3] * 1e-3;
        double wy = matsimnibs[1][0] * x + matsimnibs[1][1] * y + matsimnibs[1][2] * z + matsimnibs[1][3] * 1e-3;
        double wz = matsimnibs[2][0] * x + matsimnibs[2][1] * y + matsimnibs[2][2] * z + matsimnibs[2][3] * 1e-3;

        // Rotate magnetic dipole moment
        double wmx = matsimnibs[0][0] * mx + matsimnibs[0][1] * my + matsimnibs[0][2] * mz;
        double wmy = matsimnibs[1][0] * mx + matsimnibs[1][1] * my + matsimnibs[1][2] * mz;
        double wmz = matsimnibs[2][0] * mx + matsimnibs[2][1] * my + matsimnibs[2][2] * mz;

        dipoles.px.push_back(static_cast<float>(wx));
        dipoles.py.push_back(static_cast<float>(wy));
        dipoles.pz.push_back(static_cast<float>(wz));
        dipoles.mx.push_back(static_cast<float>(wmx));
        dipoles.my.push_back(static_cast<float>(wmy));
        dipoles.mz.push_back(static_cast<float>(wmz));
    }

    if (dipoles.size() == 0) {
        throw std::runtime_error("No valid dipoles parsed from .ccd file: " + ccd_path);
    }
    return dipoles;
}

struct VoxelField3 {
    std::vector<float> x, y, z; // Grid-axis components (V/m)
};

struct GridAxes {
    float ux[3], uy[3], uz[3];
};

GridAxes extract_grid_axes(const NiftiVolume& vol) {
    auto norm3 = [](float a, float b, float c, float out[3]) {
        float len = std::sqrt(a * a + b * b + c * c);
        float inv = (len > 0.0f) ? (1.0f / len) : 1.0f;
        out[0] = a * inv; out[1] = b * inv; out[2] = c * inv;
    };
    GridAxes ax{};
    norm3(vol.srow_x[0], vol.srow_y[0], vol.srow_z[0], ax.ux);
    norm3(vol.srow_x[1], vol.srow_y[1], vol.srow_z[1], ax.uy);
    norm3(vol.srow_x[2], vol.srow_y[2], vol.srow_z[2], ax.uz);
    return ax;
}

inline void voxel_to_world_mm(const NiftiVolume& vol, const PadInfo& pad,
                              int i_pad, int j_pad, int k_pad,
                              double& x_mm, double& y_mm, double& z_mm) {
    double vx = static_cast<double>(i_pad - pad.pad_x);
    double vy = static_cast<double>(j_pad - pad.pad_y);
    double vz = static_cast<double>(k_pad - pad.pad_z);
    x_mm = vol.srow_x[0] * vx + vol.srow_x[1] * vy + vol.srow_x[2] * vz + vol.srow_x[3];
    y_mm = vol.srow_y[0] * vx + vol.srow_y[1] * vy + vol.srow_y[2] * vz + vol.srow_y[3];
    z_mm = vol.srow_z[0] * vx + vol.srow_z[1] * vy + vol.srow_z[2] * vz + vol.srow_z[3];
}

VoxelField3 compute_A0_prime_omp(const GridLevel& fine,
                                 const NiftiVolume& vol,
                                 const PadInfo& pad,
                                 const CoilDipoles& dip,
                                 double didt) {
    const size_t n_vox = fine.sigma.size();
    VoxelField3 A0;
    A0.x.assign(n_vox, 0.0f);
    A0.y.assign(n_vox, 0.0f);
    A0.z.assign(n_vox, 0.0f);

    const GridAxes axes = extract_grid_axes(vol);
    const float scale = static_cast<float>(1e-7 * didt); // (mu0 / 4pi) * dI/dt
    const int n_dip = static_cast<int>(dip.size());
    const float* px = dip.px.data();
    const float* py = dip.py.data();
    const float* pz = dip.pz.data();
    const float* mx = dip.mx.data();
    const float* my = dip.my.data();
    const float* mz = dip.mz.data();

    #pragma omp parallel for schedule(dynamic, 2)
    for (int k = 0; k < fine.nz; ++k) {
        for (int j = 0; j < fine.ny; ++j) {
            for (int i = 0; i < fine.nx; ++i) {
                size_t v = fine.v_idx(i, j, k);
                if (fine.sigma[v] == 0.0) continue; // Skip air

                double x_mm, y_mm, z_mm;
                voxel_to_world_mm(vol, pad, i, j, k, x_mm, y_mm, z_mm);
                const float rx = static_cast<float>(x_mm * 1e-3);
                const float ry = static_cast<float>(y_mm * 1e-3);
                const float rz = static_cast<float>(z_mm * 1e-3);

                float ax = 0.0f, ay = 0.0f, az = 0.0f;

                OMP_SIMD_RED_AXAYAZ
                for (int d = 0; d < n_dip; ++d) {
                    float dx = rx - px[d];
                    float dy = ry - py[d];
                    float dz = rz - pz[d];
                    float r2 = dx * dx + dy * dy + dz * dz;
                    float inv_r3 = (r2 > 1e-12f) ? (1.0f / (r2 * std::sqrt(r2))) : 0.0f;
                    ax += (my[d] * dz - mz[d] * dy) * inv_r3;
                    ay += (mz[d] * dx - mx[d] * dz) * inv_r3;
                    az += (mx[d] * dy - my[d] * dx) * inv_r3;
                }

                ax *= scale; ay *= scale; az *= scale;

                // Project world A0' onto voxel-grid axes (i, j, k)
                A0.x[v] = ax * axes.ux[0] + ay * axes.ux[1] + az * axes.ux[2];
                A0.y[v] = ax * axes.uy[0] + ay * axes.uy[1] + az * axes.uy[2];
                A0.z[v] = ax * axes.uz[0] + ay * axes.uz[1] + az * axes.uz[2];
            }
        }
    }
    return A0;
}

void assemble_fem_rhs_omp(GridLevel& fine, const VoxelField3& A0) {
    std::fill(fine.rhs.begin(), fine.rhs.end(), 0.0);
    const double coeff = -0.25 / fine.h;

    OMP_PARALLEL_FOR_COLLAPSE2
    for (int k = 1; k < fine.nz; ++k) {
        for (int j = 1; j < fine.ny; ++j) {
            for (int i = 1; i < fine.nx; ++i) {
                double sum = 0.0;
                for (int dz = 0; dz <= 1; ++dz) {
                    const double sz = (dz == 0) ? 1.0 : -1.0;
                    for (int dy = 0; dy <= 1; ++dy) {
                        const double sy = (dy == 0) ? 1.0 : -1.0;
                        for (int dx = 0; dx <= 1; ++dx) {
                            const double sx = (dx == 0) ? 1.0 : -1.0;
                            size_t v = fine.v_idx(i - 1 + dx, j - 1 + dy, k - 1 + dz);
                            double sig = fine.sigma[v];
                            if (sig == 0.0) continue;
                            sum += sig * (sx * A0.x[v] + sy * A0.y[v] + sz * A0.z[v]);
                        }
                    }
                }
                fine.rhs[fine.n_idx(i, j, k)] = coeff * sum;
            }
        }
    }
}

/*
    4. Matrix-Free 21-Point FEM Stencil & OpenMP Geometric Multigrid Kernels
*/
inline void eval_stencil(const GridLevel& g, int i, int j, int k,
                         double& diag, double& off_diag_sum) {
    // 8 surrounding voxels (octant bits: 0->x, 1->y, 2->z; 0 means -1, 1 means 0)
    const size_t sy = static_cast<size_t>(g.nx);
    const size_t sz = static_cast<size_t>(g.nx) * g.ny;
    const size_t v0 = g.v_idx(i - 1, j - 1, k - 1);

    const double s0 = g.sigma[v0];
    const double s1 = g.sigma[v0 + 1];
    const double s2 = g.sigma[v0 + sy];
    const double s3 = g.sigma[v0 + sy + 1];
    const double s4 = g.sigma[v0 + sz];
    const double s5 = g.sigma[v0 + sz + 1];
    const double s6 = g.sigma[v0 + sz + sy];
    const double s7 = g.sigma[v0 + sz + sy + 1];

    const double s_sum = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    if (s_sum == 0.0) {
        diag = 0.0;
        off_diag_sum = 0.0;
        return;
    }

    // Center weight: (8/3) * (s_sum / 8) = s_sum / 3
    diag = s_sum * (1.0 / 3.0);

    const ptrdiff_t ny1 = static_cast<ptrdiff_t>(g.nx + 1);
    const ptrdiff_t nz1 = static_cast<ptrdiff_t>(g.nx + 1) * (g.ny + 1);
    const double* p = &g.phi[g.n_idx(i, j, k)];

    // 8 corner neighbors: weight (1/12) * s_a
    const double corner =
        s0 * p[-nz1 - ny1 - 1] + s1 * p[-nz1 - ny1 + 1] +
        s2 * p[-nz1 + ny1 - 1] + s3 * p[-nz1 + ny1 + 1] +
        s4 * p[+nz1 - ny1 - 1] + s5 * p[+nz1 - ny1 + 1] +
        s6 * p[+nz1 + ny1 - 1] + s7 * p[+nz1 + ny1 + 1];

    // 12 edge neighbors: weight (1/6) * 0.5 * (s_a + s_b) = (1/12) * (s_a + s_b)
    const double edge =
        // 4 edges parallel to x (ox = 0)
        (s0 + s1) * p[-nz1 - ny1] + (s2 + s3) * p[-nz1 + ny1] +
        (s4 + s5) * p[+nz1 - ny1] + (s6 + s7) * p[+nz1 + ny1] +
        // 4 edges parallel to y (oy = 0)
        (s0 + s2) * p[-nz1 - 1]   + (s1 + s3) * p[-nz1 + 1]   +
        (s4 + s6) * p[+nz1 - 1]   + (s5 + s7) * p[+nz1 + 1]   +
        // 4 edges parallel to z (oz = 0)
        (s0 + s4) * p[-ny1 - 1]   + (s1 + s5) * p[-ny1 + 1]   +
        (s2 + s6) * p[+ny1 - 1]   + (s3 + s7) * p[+ny1 + 1];

    off_diag_sum = (corner + edge) * (1.0 / 12.0);
}

// Race-free 8-color checkerboard SOR smoother
void sor_sweep_omp(GridLevel& g, int iterations, double omega = 1.4) {
    const double h2 = g.h * g.h;
    const size_t total_nodes = static_cast<size_t>(g.nx) * g.ny * g.nz;
    const bool use_omp = (total_nodes > 32768);

    for (int it = 0; it < iterations; ++it) {
        for (int color = 0; color < 8; ++color) {
            const int i_start = (color & 1) ? 1 : 2;
            const int j_start = ((color >> 1) & 1) ? 1 : 2;
            const int k_start = ((color >> 2) & 1) ? 1 : 2;

            #pragma omp parallel for schedule(static) if(use_omp)
            for (int k = k_start; k < g.nz; k += 2) {
                for (int j = j_start; j < g.ny; j += 2) {
                    for (int i = i_start; i < g.nx; i += 2) {
                        double diag, off_diag;
                        eval_stencil(g, i, j, k, diag, off_diag);
                        if (diag == 0.0) continue;

                        size_t idx = g.n_idx(i, j, k);
                        double phi_gs = (h2 * g.rhs[idx] + off_diag) / diag;
                        g.phi[idx] = (1.0 - omega) * g.phi[idx] + omega * phi_gs;
                    }
                }
            }
        }
    }
}

void compute_residual_omp(GridLevel& g) {
    const double inv_h2 = 1.0 / (g.h * g.h);
    const size_t total_nodes = static_cast<size_t>(g.nx) * g.ny * g.nz;
    const bool use_omp = (total_nodes > 32768);

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int k = 1; k < g.nz; ++k) {
        for (int j = 1; j < g.ny; ++j) {
            for (int i = 1; i < g.nx; ++i) {
                size_t idx = g.n_idx(i, j, k);
                double diag, off_diag;
                eval_stencil(g, i, j, k, diag, off_diag);
                if (diag == 0.0) {
                    g.res[idx] = 0.0;
                    continue;
                }
                g.res[idx] = g.rhs[idx] - inv_h2 * (diag * g.phi[idx] - off_diag);
            }
        }
    }
}

// 27-point full-weighting restriction (exact Galerkin adjoint R = (1/8) P^T)
// Uses 1D weights {1/4, 1/2, 1/4} -> 3D weights {1/8, 1/16, 1/32, 1/64}, which
// annihilates high-frequency checkerboard modes and ensures stable ~10x/cycle
// convergence with 8-color OpenMP SOR (compare {1/6, 4/6, 1/6} for serial SOR).
void restrict_residual_omp(const GridLevel& fine, GridLevel& coarse) {
    constexpr double w1d[3] = {0.25, 0.5, 0.25};
    std::fill(coarse.rhs.begin(), coarse.rhs.end(), 0.0);
    const size_t total_coarse = static_cast<size_t>(coarse.nx) * coarse.ny * coarse.nz;
    const bool use_omp = (total_coarse > 16384);

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int ck = 1; ck < coarse.nz; ++ck) {
        for (int cj = 1; cj < coarse.ny; ++cj) {
            for (int ci = 1; ci < coarse.nx; ++ci) {
                double sum = 0.0;
                for (int dz = -1; dz <= 1; ++dz) {
                    const double wz = w1d[dz + 1];
                    for (int dy = -1; dy <= 1; ++dy) {
                        const double wyz = wz * w1d[dy + 1];
                        for (int dx = -1; dx <= 1; ++dx) {
                            sum += (wyz * w1d[dx + 1]) *
                                   fine.res[fine.n_idx(2 * ci + dx, 2 * cj + dy, 2 * ck + dz)];
                        }
                    }
                }
                coarse.rhs[coarse.n_idx(ci, cj, ck)] = sum;
            }
        }
    }
}

// Trilinear prolongation and correction update
void prolongate_and_add_omp(const GridLevel& coarse, GridLevel& fine) {
    const size_t total_fine = static_cast<size_t>(fine.nx) * fine.ny * fine.nz;
    const bool use_omp = (total_fine > 32768);

    #pragma omp parallel for schedule(static) if(use_omp)
    for (int fk = 1; fk < fine.nz; ++fk) {
        const int ck = fk >> 1, oz = fk & 1;
        const double wz = oz ? 0.5 : 1.0;
        for (int fj = 1; fj < fine.ny; ++fj) {
            const int cj = fj >> 1, oy = fj & 1;
            const double wyz = wz * (oy ? 0.5 : 1.0);
            for (int fi = 1; fi < fine.nx; ++fi) {
                const int ci = fi >> 1, ox = fi & 1;
                const double w = wyz * (ox ? 0.5 : 1.0);

                double interp = 0.0;
                for (int dz = 0; dz <= oz; ++dz) {
                    for (int dy = 0; dy <= oy; ++dy) {
                        for (int dx = 0; dx <= ox; ++dx) {
                            interp += coarse.phi[coarse.n_idx(ci + dx, cj + dy, ck + dz)];
                        }
                    }
                }
                fine.phi[fine.n_idx(fi, fj, fk)] += w * interp;
            }
        }
    }
}

double l2_norm_omp(const std::vector<double>& v) {
    double sum = 0.0;
    const long long n = static_cast<long long>(v.size());
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 0; i < n; ++i) {
        sum += v[static_cast<size_t>(i)] * v[static_cast<size_t>(i)];
    }
    return std::sqrt(sum);
}

void v_cycle_omp(std::vector<GridLevel>& levels, int l) {
    GridLevel& fine = levels[l];
    sor_sweep_omp(fine, 3, 1.4);
    compute_residual_omp(fine);

    GridLevel& coarse = levels[l + 1];
    restrict_residual_omp(fine, coarse);
    std::fill(coarse.phi.begin(), coarse.phi.end(), 0.0);

    if (l + 1 == static_cast<int>(levels.size()) - 1) {
        sor_sweep_omp(coarse, 100, 1.4); // Coarsest grid level
    } else {
        v_cycle_omp(levels, l + 1);
    }

    prolongate_and_add_omp(coarse, fine);

    if (l > 0) {
        sor_sweep_omp(fine, 1, 1.4); // Skipped on finest level (l == 0)
    }
}

/*
    5. Electric Field Recover: E = -grad(phi) - A0.
*/
struct EFieldOutput {
    std::vector<float> mag; // 3D |E| (nx * ny * nz)
    std::vector<float> vec; // 4D RAS+ world vector E: [Ex..., Ey..., Ez...] (3 * nx * ny * nz)
};

EFieldOutput compute_efield_unpadded(const GridLevel& fine,
                                     const VoxelField3& A0,
                                     const NiftiVolume& vol,
                                     const PadInfo& pad,
                                     bool want_vector) {
    const size_t n_orig = static_cast<size_t>(vol.nx) * vol.ny * vol.nz;
    EFieldOutput out;
    out.mag.assign(n_orig, 0.0f);
    if (want_vector) {
        out.vec.assign(3 * n_orig, 0.0f);
    }

    const GridAxes axes = extract_grid_axes(vol);
    const double inv_4h = 0.25 / fine.h;

    OMP_PARALLEL_FOR_COLLAPSE2
    for (int k = 0; k < vol.nz; ++k) {
        for (int j = 0; j < vol.ny; ++j) {
            for (int i = 0; i < vol.nx; ++i) {
                const size_t orig_idx = (static_cast<size_t>(k) * vol.ny + j) * vol.nx + i;
                if (vol.labels[orig_idx] == 0) continue; // Mask outside head

                const int ip = i + pad.pad_x;
                const int jp = j + pad.pad_y;
                const int kp = k + pad.pad_z;

                const double p000 = fine.phi[fine.n_idx(ip,     jp,     kp    )];
                const double p100 = fine.phi[fine.n_idx(ip + 1, jp,     kp    )];
                const double p010 = fine.phi[fine.n_idx(ip,     jp + 1, kp    )];
                const double p110 = fine.phi[fine.n_idx(ip + 1, jp + 1, kp    )];
                const double p001 = fine.phi[fine.n_idx(ip,     jp,     kp + 1)];
                const double p101 = fine.phi[fine.n_idx(ip + 1, jp,     kp + 1)];
                const double p011 = fine.phi[fine.n_idx(ip,     jp + 1, kp + 1)];
                const double p111 = fine.phi[fine.n_idx(ip + 1, jp + 1, kp + 1)];

                const double dphidx = inv_4h * ((p100 - p000) + (p110 - p010) + (p101 - p001) + (p111 - p011));
                const double dphidy = inv_4h * ((p010 - p000) + (p110 - p100) + (p011 - p001) + (p111 - p101));
                const double dphidz = inv_4h * ((p001 - p000) + (p101 - p100) + (p011 - p010) + (p111 - p110));

                const size_t v = fine.v_idx(ip, jp, kp);
                // Grid-axis E components (along voxel i, j, k axes)
                const double ex_g = -dphidx - A0.x[v];
                const double ey_g = -dphidy - A0.y[v];
                const double ez_g = -dphidz - A0.z[v];

                out.mag[orig_idx] = static_cast<float>(std::sqrt(ex_g * ex_g + ey_g * ey_g + ez_g * ez_g));

                if (want_vector) {
                    // Transform back to world RAS+ coordinates: E_w = ex_g*ux + ey_g*uy + ez_g*uz
                    const double ex_w = ex_g * axes.ux[0] + ey_g * axes.uy[0] + ez_g * axes.uz[0];
                    const double ey_w = ex_g * axes.ux[1] + ey_g * axes.uy[1] + ez_g * axes.uz[1];
                    const double ez_w = ex_g * axes.ux[2] + ey_g * axes.uy[2] + ez_g * axes.uz[2];

                    out.vec[orig_idx]              = static_cast<float>(ex_w);
                    out.vec[orig_idx + n_orig]     = static_cast<float>(ey_w);
                    out.vec[orig_idx + 2 * n_orig] = static_cast<float>(ez_w);
                }
            }
        }
    }
    return out;
}
/*
    6. CLI Entry Points
*/
static void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " <final_tissues_MNI.nii> <coil.ccd> <out_efield.nii> [options]\n"
              << "Options:\n"
              << "  --matsimnibs <file>   4x4 coil placement matrix (mm in 4th column)\n"
              << "  --didt <val>          dI/dt in A/s (default: 1e6)\n"
              << "  --levels <L>          Multigrid levels (default: 4)\n"
              << "  --tol <val>           Relative residual tolerance (default: 1e-5)\n"
              << "  --max-iter <N>        Maximum V-cycles (default: 30)\n"
              << "  --no-smooth           Skip 3x3x3 conductivity smoothing\n"
              << "  --out-vec <file>      Save 4D world-space (RAS+) vector E (nx,ny,nz,3)\n";
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string nifti_path = argv[1];
    const std::string ccd_path   = argv[2];
    const std::string out_path   = argv[3];
    std::string out_vec_path;

    std::string mat_path;
    double didt = 1e6;       // 1 A/us (SimNIBS default)
    int num_levels = 4;      // 4 levels for 1 mm / 2 mm grids
    double tol = 1e-5;       // <1% E-field truncation error
    int max_iter = 30;
    bool do_smooth = true;

    for (int a = 4; a < argc; ++a) {
        std::string arg = argv[a];
        if (arg == "--matsimnibs" && a + 1 < argc)      mat_path = argv[++a];
        else if (arg == "--didt" && a + 1 < argc)       didt = std::stod(argv[++a]);
        else if (arg == "--levels" && a + 1 < argc)     num_levels = std::stoi(argv[++a]);
        else if (arg == "--tol" && a + 1 < argc)        tol = std::stod(argv[++a]);
        else if (arg == "--max-iter" && a + 1 < argc)   max_iter = std::stoi(argv[++a]);
        else if (arg == "--no-smooth")                  do_smooth = false;
        else if (arg == "--out-vec" && a + 1 < argc)    out_vec_path = argv[++a];
        else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    try {
        using Clock = std::chrono::steady_clock;
        auto elapsed_s = [](Clock::time_point t0) {
            return std::chrono::duration<double>(Clock::now() - t0).count();
        };

#ifdef _OPENMP
        std::cout << "OpenMP enabled: max threads = " << omp_get_max_threads() << "\n";
#else
        std::cout << "OpenMP disabled (single-threaded build)\n";
#endif

        // 1. Load charm NIfTI
        auto t0 = Clock::now();
        NiftiVolume vol = load_charm_nifti(nifti_path);
        std::cout << "Loaded NIfTI: " << vol.nx << "x" << vol.ny << "x" << vol.nz
                  << " | spacing = (" << vol.dx_m * 1e3 << ", " << vol.dy_m * 1e3
                  << ", " << vol.dz_m * 1e3 << ") mm [" << std::fixed << std::setprecision(2)
                  << elapsed_s(t0) << " s]\n";

        if (std::abs(vol.dx_m - vol.dy_m) > 1e-5 || std::abs(vol.dx_m - vol.dz_m) > 1e-5) {
            std::cerr << "Warning: Voxels are not strictly cubical; FEM stencil assumes isotropic h = dx.\n";
        }

        // 2. Initialize multigrid hierarchy, smooth conductivity, coarsen
        t0 = Clock::now();
        PadInfo pad{};
        auto levels = init_hierarchy_from_nifti(vol, num_levels, pad);
        if (do_smooth) {
            smooth_conductivity_3x3x3(levels[0]);
        }
        build_coarse_conductivities(levels);
        std::cout << "Multigrid hierarchy (" << num_levels << " levels): finest = "
                  << levels[0].nx << "x" << levels[0].ny << "x" << levels[0].nz
                  << ", coarsest = " << levels.back().nx << "x" << levels.back().ny
                  << "x" << levels.back().nz << " [" << elapsed_s(t0) << " s]\n";

        // 3. Load .ccd coil & transform with matsimnibs
        Mat4 matsimnibs = {{
            {{ 1.0,  0.0,  0.0,   0.0 }},
            {{ 0.0, -1.0,  0.0,   0.0 }},
            {{ 0.0,  0.0, -1.0, 105.0 }}, // Default: above MNI vertex (0, 0, 105 mm)
            {{ 0.0,  0.0,  0.0,   1.0 }}
        }};
        if (!mat_path.empty()) {
            matsimnibs = load_matsimnibs(mat_path);
        } else {
            std::cout << "No --matsimnibs provided; using default vertex coil placement (0, 0, 105 mm).\n";
        }

        CoilDipoles dipoles = load_ccd_file(ccd_path, matsimnibs);
        std::cout << "Loaded " << dipoles.size() << " magnetic dipoles from " << ccd_path << "\n";

        // 4. Compute A0' and assemble FEM source term f^h
        t0 = Clock::now();
        VoxelField3 A0 = compute_A0_prime_omp(levels[0], vol, pad, dipoles, didt);
        assemble_fem_rhs_omp(levels[0], A0);
        const double rhs_norm = l2_norm_omp(levels[0].rhs);
        std::cout << "Computed A0' and assembled FEM RHS (||f||_2 = " << std::scientific
                  << std::setprecision(3) << rhs_norm << ") [" << std::fixed
                  << std::setprecision(2) << elapsed_s(t0) << " s]\n";

        if (rhs_norm == 0.0) {
            throw std::runtime_error("Zero RHS norm (check tissue labels and coil placement).");
        }

        // 5. Solve with geometric multigrid V-cycles
        std::fill(levels[0].phi.begin(), levels[0].phi.end(), 0.0);
        auto t_solve = Clock::now();
        for (int iter = 1; iter <= max_iter; ++iter) {
            auto t_cyc = Clock::now();
            v_cycle_omp(levels, 0);
            compute_residual_omp(levels[0]);
            double rel_res = l2_norm_omp(levels[0].res) / rhs_norm;
            std::cout << "  V-cycle " << std::setw(2) << iter
                      << " | rel_res = " << std::scientific << std::setprecision(3) << rel_res
                      << " | time = " << std::fixed << std::setprecision(3) << elapsed_s(t_cyc) << " s\n";
            if (rel_res < tol) break;
        }
        std::cout << "Multigrid solve finished in " << std::fixed << std::setprecision(2)
                  << elapsed_s(t_solve) << " s\n";

        // 6. Compute |E| (and optionally RAS+ vector E) and save NIfTI
        t0 = Clock::now();
        const bool want_vec = !out_vec_path.empty();
        auto efield = compute_efield_unpadded(levels[0], A0, vol, pad, want_vec);
        float max_e = efield.mag.empty() ? 0.0f
                      : *std::max_element(efield.mag.begin(), efield.mag.end());
        save_nifti_float32(out_path, vol.hdr, efield.mag, 1);
        std::cout << "Saved |E| to " << out_path << " (max |E| = " << std::fixed
                  << std::setprecision(3) << max_e << " V/m) [" << elapsed_s(t0) << " s]\n";

        if (want_vec) {
            t0 = Clock::now();
            save_nifti_float32(out_vec_path, vol.hdr, efield.vec, 3);
            std::cout << "Saved vector E (RAS+) to " << out_vec_path
                      << " [" << std::fixed << std::setprecision(2) << elapsed_s(t0) << " s]\n";
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
