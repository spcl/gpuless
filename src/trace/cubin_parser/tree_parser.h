#ifndef GPULESS_TREE_PARSER_H
#define GPULESS_TREE_PARSER_H

#include <array>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../cubin_analysis.hpp"
#include "ptx_tree.h"

namespace PtxTreeParser {

class TreeParser {
  private:
    std::unordered_set<std::string> _param_names;

    PtxOperand parseArgument(std::string arg);
    PtxNodeKind parseOperation(const std::string_view &op, int64_t &vec_op);

  public:
    explicit TreeParser(std::unordered_set<std::string> param_names) : _param_names(std::move(param_names)) {}

    std::vector<std::pair<std::unique_ptr<PtxTree>, std::string>>
    parsePtxTrees(std::string_view ss);
};

} // namespace PtxTreeParser

#endif // GPULESS_TREE_PARSER_H