#ifndef GPULESS_SHMEM_MEMPOOL_HPP
#define GPULESS_SHMEM_MEMPOOL_HPP

#include <queue>
#include <iostream>
#include <unordered_map>

#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <spdlog/spdlog.h>

namespace gpuless {

struct MemChunk {

    // Standard size of a memory chunk
    static constexpr int CHUNK_SIZE = 128 * 1024 * 1024;
    void* ptr;
    std::string name;
    size_t size;

    void allocate(size_t size = CHUNK_SIZE)
    {
        //int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0600);
        int fd = shm_open(name.c_str(), O_CREAT | O_RDWR, 0600);
        if(fd == -1) {
          spdlog::error("Fatal error in shm_open! {}", strerror(errno));
          abort();
        }

        int ret  = ftruncate(fd, size);
        if(ret == -1) {
          spdlog::error("Fatal error in ftruncate! {}", strerror(errno));
          abort();
        }
        //std::cerr << fd << " " << " " << size << " " << ret << " " << errno << std::endl;

        ptr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if(ptr == (void*)-1) {
          spdlog::error("Fatal error in mmap! {}", strerror(errno));
          abort();
        }
        //std::cerr << "allocate " << name << " " << size << " " << errno << " " << reinterpret_cast<std::uintptr_t>(ptr) << std::endl;

        this->size = size;
    }

    void open()
    {
        //std::cerr << "open " << name << std::endl;
        int fd = shm_open(name.c_str(), O_RDWR, 0);
        ptr = mmap(NULL, CHUNK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    }

    void close()
    {
        //std::cerr << "Close " << name << std::endl;
        munmap(ptr, CHUNK_SIZE);
        //shm_unlink(name.c_str());
    }
};

class MemPoolRead {
    //std::queue<MemChunk> used_chunks;
    std::unordered_map<std::string, void*> used_chunks;
    // List of owned memory chunks that need to be released.
    std::vector<std::string> names;
public:

    void* get(const std::string& name)
    {
      auto it = used_chunks.find(name);
      if(it == used_chunks.end()) {

        int fd = shm_open(name.c_str(), O_RDWR, 0);
        if(fd == -1) {
          spdlog::error("Fatal error of {} in shm_open! {}", name, strerror(errno));
          abort();
        }

        struct stat st;
        fstat(fd, &st);

        //auto ptr = mmap(NULL, MemChunk::CHUNK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        auto ptr = mmap(NULL, st.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        //std::cerr << "open " << name << " " << fd << " " << ptr << std::endl;
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

    //std::queue<MemChunk> chunks;
    std::vector<MemChunk> user_chunks;

    // This one is for memory pages given to mignificient operations. These are transmitted as page names
    std::unordered_map<std::string, MemChunk> used_chunks;

    // Used to quickly map a pointer to the shared memory page.
    std::unordered_map<const void*, MemChunk> borrowed_chunks;

    int counter = 0;

    std::string _user_name;

public:

    void set_user_name(const char* _user_name)
    {
      this->_user_name = _user_name;
    }

    void give(const std::string& name)
    {
      SPDLOG_INFO("Give back chunk {}", name);
      bool inserted = false;
      auto it = used_chunks.find(name);
      if(it == used_chunks.end()) {
        spdlog::error("Cannot find the address of page with name {}", name);
        abort();
      }
      for(int i = 0; i < user_chunks.size(); ++i) {
        if(user_chunks[i].ptr == nullptr) {
          user_chunks[i] = (*it).second;
          inserted = true;
          break;
        }
      }
      if(!inserted)
        user_chunks.push_back((*it).second);
    }

    // Only to return MIGnificient mallocs
    void give(void *ptr)
    {
      auto it = borrowed_chunks.find(ptr);
      if(it == borrowed_chunks.end()) {
        spdlog::error("Cannot find the chunk allocation of address {}", fmt::ptr(ptr));
        abort();
      }
      bool inserted = false;
      for(int i = 0; i < user_chunks.size(); ++i) {
        if(user_chunks[i].ptr == nullptr) {
          user_chunks[i] = (*it).second;
          inserted = true;
          break;
        }
      }
      if(!inserted)
        user_chunks.push_back((*it).second);
    }

    // Used to verify if the provided memory address is our mapped chunk
    std::optional<std::string> get_name(const void *ptr)
    {
      auto it = borrowed_chunks.find(ptr);
      if(it == borrowed_chunks.end()) {
        return std::optional<std::string>{};
      }
      return (*it).second.name;
    }

    MemChunk get(size_t size, bool malloc_allocation = false)
    {
      size_t q_size = user_chunks.size();
      int pos = -1;
      int min_size = INT_MAX;

      // Find first chunk with minimum size
      for(int i = 0; i < q_size; ++i) {

        if(user_chunks[i].ptr != nullptr && user_chunks[i].size >= size) {
          if(user_chunks[i].size < min_size) {
            pos = i;
            min_size = user_chunks[i].size;
          }
        }
      }

      if(pos != -1) {

        MemChunk chunk = user_chunks[pos];
        user_chunks[pos].ptr = nullptr;
        return chunk;

      } else {

        std::string name = fmt::format("/gpuless_{}_{}", _user_name, counter++);
        SPDLOG_INFO("Create new chunk {}", name);
        MemChunk chunk{nullptr, name};
        chunk.allocate(size);
        // We don't insert the chunk to the list of user chunks -> it is taken anyway
        if(malloc_allocation) {
          borrowed_chunks[chunk.ptr] = chunk;
        } else {
          used_chunks[name] = chunk;
        }

        return chunk;
      }
    }

    ~MemPool()
    {
      for(auto [name, chunk] : used_chunks)
        chunk.close();
      for(auto [ptr, chunk] : borrowed_chunks)
        chunk.close();
    }
};

}

#endif
