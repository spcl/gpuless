#ifndef __CUBIN_ANALYSIS_HPP__
#define __CUBIN_ANALYSIS_HPP__

#include <filesystem>
#include <map>
#include <string>
#include <vector>

enum PtxParameterType {
    s8 = 0,
    s16 = 1,
    s32 = 2,
    s64 = 3, // signed integers
    u8 = 4,
    u16 = 5,
    u32 = 6,
    u64 = 7, // unsigned integers
    f16 = 8,
    f16x2 = 9,
    f32 = 10,
    f64 = 11, // floating-point
    b8 = 12,
    b16 = 13,
    b32 = 14,
    b64 = 15,     // untyped bits
    pred = 16,    // predicate
    invalid = 17, // invalid type for signaling errors
};

std::map<std::string, PtxParameterType> &getStrToPtxParameterType();
std::map<PtxParameterType, std::string> &getPtxParameterTypeToStr();
std::map<PtxParameterType, int> &getPtxParameterTypeToSize();

struct PTXKParamInfo {
    std::string paramName;
    PtxParameterType type;
    int typeSize;
    int align;
    int size;
};

struct ELFKParamInfo {
    int size;
};

class CubinAnalyzerPTX {
  private:
    bool initialized_ = false;
    std::map<std::string, std::vector<PTXKParamInfo>> kernel_to_kparaminfos;

    static PtxParameterType ptxParameterTypeFromString(const std::string &str);
    static int byteSizePtxParameterType(PtxParameterType type);

    bool isCached(const std::filesystem::path &fname);
    bool loadAnalysisFromCache(const std::filesystem::path &fname);
    void storeAnalysisToCache(
        const std::filesystem::path &fname,
        const std::map<std::string, std::vector<PTXKParamInfo>> &data);

    std::vector<PTXKParamInfo> parsePtxParameters(const std::string &params);
    bool analyzePtx(const std::filesystem::path &path, int major_version,
                    int minor_version);
    static size_t pathToHash(const std::filesystem::path &path);

  public:
    CubinAnalyzerPTX() = default;
    bool isInitialized();
    bool analyze(const std::vector<std::string>& cuda_binaries, int major_version,
                 int minor_version);

    bool kernel_parameters(std::string &kernel,
                           std::vector<PTXKParamInfo> &params) const;
    bool kernel_module(std::string &kernel, std::vector<uint8_t> &module_data);
};

class CubinAnalyzerELF {
  private:
    bool initialized_ = false;
    std::map<std::string, std::vector<int>> kernel_to_kparaminfos;

    std::vector<ELFKParamInfo> parseElfParameters(const std::string &params);
    bool analyzeELF(const std::filesystem::path &path, const std::string& compute_version);

  public:
    CubinAnalyzerELF() = default;
    bool isInitialized();
    bool analyze(const std::vector<std::string>& cuda_binaries, const std::string& compute_version);
    void storeAnalysisToCache(const std::filesystem::path &fname);
    bool loadAnalysisFromCache(const std::filesystem::path &fname);

    bool kernel_parameters(std::string &kernel, std::vector<int> &params) const;
};

#endif // __CUBIN_ANALYSIS_HPP__
