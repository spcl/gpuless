#ifndef GPULESS_SHMEM_MEMPOOL_HPP
#define GPULESS_SHMEM_MEMPOOL_HPP

#include <iostream>
#include <queue>
#include <unordered_map>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/types.h>

#include <spdlog/spdlog.h>

namespace gpuless {

struct MemChunk {

    // Standard size of a memory chunk
    static constexpr int CHUNK_SIZE = 128 * 1024 * 1024;
    void* ptr;
    std::string name;

    void allocate()
    {
        //int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
        std::cerr << fd << " " << errno << std::endl;
        int ret  = ftruncate(fd, CHUNK_SIZE);
        std::cerr << fd << " " << ret << " " << errno << std::endl;

        ptr = mmap(NULL, CHUNK_SIZE, PROT_READ | PROT_WRITE,
                    MAP_SHARED, fd, 0);
        std::cerr << "allocate " << name << " " << CHUNK_SIZE << " " << reinterpret_cast<std::uintptr_t>(ptr) << std::endl;
    }

    void open()
    {
        std::cerr << "open " << name << std::endl;
        int fd = shm_open(name.c_str(), O_RDWR, 0);
        ptr = mmap(NULL, CHUNK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }

    void close()
    {
        std::cerr << "Close " << name << std::endl;
        munmap(ptr, CHUNK_SIZE);
        //shm_unlink(name.c_str());
    }
};

class MemPoolRead {
    //std::queue<MemChunk> used_chunks;
    std::unordered_map<std::string, void*> used_chunks;
    std::vector<std::string> names;
public:

    void* get(const std::string& name)
    {
      auto it = used_chunks.find(name);
      if(it == used_chunks.end()) {

        int fd = shm_open(name.c_str(), O_RDWR, 0);
        auto ptr = mmap(NULL, MemChunk::CHUNK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        std::cerr << "open " << name << " " << fd << " " << ptr << std::endl;
        used_chunks[name] = ptr;
        names.emplace_back(name);
        return ptr;
      } else {
        //std::cerr << "return " << name << " " << (*it).second << std::endl;
        return (*it).second;
      }
    }

    void close()
    {
      for(auto & name: names) {
        shm_unlink(name.c_str());
      }
    }

    static MemPoolRead& get_instance()
    {
      static MemPoolRead readers;
      return readers;
    }
};

class MemPool {

    std::queue<MemChunk> chunks;
    std::unordered_map<std::string, void*> used_chunks;

    int counter = 0;

public:

    void give(const std::string& name)
    {
      //std::cerr << "Return " << name << std::endl;
      chunks.push(MemChunk{used_chunks[name], name});
    }

    MemChunk get()
    {

      if(chunks.empty()) {

        /// FIXME: name
        std::string name = fmt::format("/gpuless_{}", counter++);
        MemChunk chunk{nullptr, name};
        chunk.allocate();
        chunks.push(chunk);

      }

      MemChunk ret = chunks.front(); 
      chunks.pop();
      used_chunks[ret.name] = ret.ptr;

      return ret;
    }

    ~MemPool()
    {
      while(!chunks.empty()) {

        MemChunk ret = chunks.front(); 
        chunks.pop();
        ret.close();

      }
    }
};

}

#endif
