#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <sys/stat.h>
#include <filesystem>

#include "../include/utils.h"

void ClearDirectory(const std::string& path) {
#if __cplusplus >= 201703L
  namespace fs = std::filesystem;
  try {
    fs::remove_all(path);  
  } catch (const fs::filesystem_error& e) {
    std::cerr << "Error deleting directory: " << e.what() << std::endl;
  }
#else
  ClearDirectoryLegacy(path)
#endif
}

void ClearDirectoryLegacy(const std::string& path) {
#if defined(_WIN32)
  std::string cmd = "rmdir /s /q \"" + path + "\"";
  system(cmd.c_str());
#else
  std::string cmd = "rm -rf \"" + path + "\"";
  system(cmd.c_str());
#endif
}

void CreateDirectory(const std::string& path) {
#if defined(_WIN32)
  mkdir(path.c_str());
#else
  mkdir(path.c_str(), 0777);
#endif
}

std::vector<char> SerializeStringSet(const std::set<std::string>& s) {
  std::vector<char> buffer;
  for (const auto& str : s) {
    buffer.insert(buffer.end(), str.begin(), str.end());
    buffer.push_back('\0');
  }
  return buffer;
}

std::set<std::string> DeserializeStringSet(const char* buffer, int size) {
  std::set<std::string> result;
  const char* ptr = buffer;
  while (ptr < buffer + size) {
    std::string s(ptr);
    result.insert(s);
    ptr += s.size() + 1;
  }
  return result;
}