#include <string>

#include "trace/cubin_analysis.hpp"

void string_split(std::string const &str, const char delim,
                  std::vector<std::string> &out)
{
  size_t start;
  size_t end = 0;
  while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
      end = str.find(delim, start);
      out.push_back(str.substr(start, end - start));
  }
}

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
