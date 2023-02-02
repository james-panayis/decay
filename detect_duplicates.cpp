#include "root.hpp"
#include "threading.hpp"

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <mutex>
#include <array>
#include <algorithm>


int main()
{
  std::vector<std::pair<std::int64_t, std::size_t>> data;

  constexpr auto filenames = std::to_array({    "../data/Lb2pKmm_mgUp_2016_UID.root",
                                            "../data/Lb2pKmm_sim_mgUp_2016_UID.root",
                                                "../data/Lb2pKmm_mgDn_2016_UID.root",
                                            "../data/Lb2pKmm_sim_mgDn_2016_UID.root",
                                                "../data/Lb2pKmm_mgUp_2017_UID.root",
                                            "../data/Lb2pKmm_sim_mgUp_2017_UID.root",
                                                "../data/Lb2pKmm_mgDn_2017_UID.root",
                                            "../data/Lb2pKmm_sim_mgDn_2017_UID.root",
                                                "../data/Lb2pKmm_mgUp_2018_UID.root",
                                            "../data/Lb2pKmm_sim_mgUp_2018_UID.root",
                                                "../data/Lb2pKmm_mgDn_2018_UID.root",
                                            "../data/Lb2pKmm_sim_mgDn_2018_UID.root"});

  std::mutex data_mutex;

  auto read_from_file = [&](const std::size_t fileno)
  {
    fmt::print("processing file {}\n", filenames[fileno]);

    const movency::root::file file(filenames[fileno]);

    auto temp = file.uncompress<std::int64_t>("UID");

    std::ranges::sort(temp);

    std::scoped_lock lock(data_mutex);

    for (auto val : temp)
      data.emplace_back(val, fileno);

    std::ranges::inplace_merge(data, data.end() - static_cast<std::ptrdiff_t>(temp.size()), {}, [](auto v){ return v.first; });
  };

  loop_threaded(read_from_file, filenames.size());

  fmt::print("checking for duplicates\n");

  auto exit_code = EXIT_SUCCESS;

  for (std::size_t i = 0; i + 1 < data.size(); ++i)
  {
    if (data[i].first == data[i + 1].first)
    {
      if (data[i].second == data[i + 1].second)
        fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "The value {} appears more than once in file {}\n",   data[i].first, filenames[data[i].second]);
      else
        fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "The value {} appears in both file {} and file {}\n", data[i].first, filenames[data[i].second], filenames[data[i + 1].second]);

      exit_code = EXIT_FAILURE;
    }
  }

  if (exit_code == EXIT_SUCCESS)
    fmt::print(fg(fmt::color::green), "The data contains no duplicates\n");

  return exit_code;
}
