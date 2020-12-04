// Check if std::filesystem is available
// Source: https://stackoverflow.com/a/54290906
// with modifications

#include <filesystem>

int main( int /*argc*/, char ** /*argv[]*/ ) {
  std::filesystem::path somepath{ "dir1/dir2/filename.txt" };
  auto fname = somepath.filename( );
  return 0;
}
