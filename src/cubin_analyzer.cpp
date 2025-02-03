#include <string>

#include "trace/cubin_analysis.hpp"
#include "utils.hpp"

int main(int argc, char** argv)
{
  std::string output{argv[1]};
  std::string compute_version{argv[2]};

  CubinAnalyzerELF analyzer;

  std::vector<std::string> cuda_binaries;
  string_split(std::string(argv[3]), ',', cuda_binaries);
  analyzer.analyze(cuda_binaries, compute_version);

  analyzer.storeAnalysisToCache(output);

  return 0;
}
