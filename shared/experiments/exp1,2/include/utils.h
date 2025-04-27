#ifndef UTILS
#define UTILS

#include <set>
#include <vector>
#include <string>

void ClearDirectory(const std::string& path);

void CreateDirectory(const std::string& path);

std::vector<char> SerializeStringSet(const std::set<std::string>& s);

std::set<std::string> DeserializeStringSet(const char* buffer, int size);

#endif