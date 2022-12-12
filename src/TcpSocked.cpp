#include "TcpSocked.hpp"
#include "utils.hpp"
#include <netinet/in.h>
#include <spdlog/spdlog.h>

TcpSocked::TcpSocked(const sockaddr_in &addr) {
    if ((m_sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        SPDLOG_ERROR("failed to open socket");
        throw "failed to open socket";
    }

    if (connect(m_sockfd, (sockaddr *)&addr, sizeof(addr)) < 0) {
        SPDLOG_ERROR("failed to connect");
        throw "failed to connect";
    }
}

TcpSocked::TcpSocked(const char *ip, const short port) {

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (inet_pton(AF_INET, ip, &addr.sin_addr) < 0) {
        SPDLOG_ERROR("Invalid IP address: {}", ip);
        throw "Invalid IP address";
    }

    if ((m_sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        SPDLOG_ERROR("failed to open socket");
        throw "failed to open socket";
    }

    if (connect(m_sockfd, (sockaddr *)&addr, sizeof(addr)) < 0) {
        SPDLOG_ERROR("failed to connect");
        throw "failed to connect";
    }
}

TcpSocked::~TcpSocked() { close(m_sockfd); }

void TcpSocked::send(const uint8_t *buf, size_t len) {
    send_buffer(m_sockfd, buf, len);
}

std::vector<uint8_t> TcpSocked::recv() { return recv_buffer(m_sockfd); }