#include "root.hpp"

#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"

#include "timeblit/random.hpp"

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <memory>
#include <thread>
#include <mutex>
#include <array>
#include <algorithm>
#include <fstream>
#include <charconv>
#include <iostream>
#include <barrier>

using namespace movency;

// x to the power of non-negative integer P
template<std::size_t P>
[[nodiscard]] constexpr auto pow(auto x) -> decltype(x)
{
  if constexpr (P == 0)
    return 1;
  else if constexpr (P % 2 == 0)
    return pow<P/2>(x) * pow<P/2>(x);
  else
    return x * pow<P/2>(x) * pow<P/2>(x);
}

//const std::size_t thread_count{std::max(1u, std::thread::hardware_concurrency() - 1)};
const std::size_t thread_count{4};

using namespace std::literals;

//constexpr auto variable_names = std::to_array<std::string_view>({"Lres_IPCHI2_OWNPV"sv, "h1_P"sv, "h1_PT"sv, "h2_P"sv, "h2_PT"sv, "Lres_FD_OWNPV"sv, "Jpsi_FD_OWNPV"sv, "Lres_TAUCHI2"sv, "Lb_IP_OWNPV"sv, "Jpsi_P"sv, "Jpsi_ENDVERTEX_CHI2"sv, "Lres_ENDVERTEX_CHI2"sv});
constexpr auto variable_names = std::to_array<std::string_view>({"Lres_IPCHI2_OWNPV"sv, "h1_P"sv, "h1_PT"sv, "h2_P"sv, "h2_PT"sv, "Lres_FD_OWNPV"sv, "Jpsi_FD_OWNPV"sv, "Lres_TAUCHI2"sv, "Lb_IP_OWNPV"sv, "Jpsi_P"sv, "Jpsi_ENDVERTEX_CHI2"sv, "Lres_ENDVERTEX_CHI2"sv, "Lb_M"sv});
//constexpr auto variable_names = std::to_array<std::string_view>({"h1_PT"sv, "h2_P"sv, "Lb_M"sv});

constexpr std::size_t full_variable_count{variable_names.size()};
constexpr std::size_t      variable_count{full_variable_count - 1};

auto create_data()
{
  movency::root::file r_real{"../data/Lb2pKmm_mgUp_2018.root"};
  movency::root::file r_simu{"../data/Lb2pKmm_sim_mgUp_2018.root"};

  std::vector<std::array<double, full_variable_count + 1>> data{};

  for (std::size_t variable_index = 0; variable_index < full_variable_count; ++variable_index)
  {
    auto variables{r_real.uncompress<double>(std::string(variable_names[variable_index]).c_str())};

    fmt::print("reading {} real {} values\n", variables.size(), variable_names[variable_index]);

    if (variables.size() > data.size())
    {
      if (data.size() == 0)
        data.resize(variables.size());
      else
      {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: (real) {} {} {}\n", data.size(), variables.size(), variable_names[variable_index]);

        std::exit(EXIT_FAILURE);
      }
    }

    for (std::size_t i = 0; i < variables.size(); ++i)
    {
      data[i][variable_index] = variables[i];
    }
  }

  std::size_t real_count = data.size();

  for (auto& record : data)
    record[full_variable_count] = 0;

  for (std::size_t variable_index = 0; variable_index < full_variable_count; ++variable_index)
  {
    auto variables{r_simu.uncompress<double>(std::string(variable_names[variable_index]).c_str())};

    fmt::print("reading {} simulated {} values\n", variables.size(), variable_names[variable_index]);

    if (variables.size() + real_count > data.size())
    {
      if (data.size() == real_count)
        data.resize(real_count + variables.size());
      else
      {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: (simu) {} {} {} {}\n", data.size(), variables.size(), variable_names[variable_index], real_count);

        std::exit(EXIT_FAILURE);
      }
    }

    for (std::size_t i = 0; i < variables.size(); ++i)
    {
      data[real_count + i][variable_index] = variables[i];
    }
  }

  for (std::size_t i = real_count; i < data.size(); ++i)
    data[i][full_variable_count] = 1;

  std::erase_if(data, [](const auto e){ return (std::abs(e[full_variable_count - 1] - 5619.60) < 300.0 ) != e[full_variable_count]; });

  real_count = 0;

  for (const auto& e : data)
    if (!e[full_variable_count])
      ++real_count;

  std::ranges::shuffle(data, movency::random::prng_);

  for (std::size_t i = 0; i < variable_count; ++i)
  {
    const double div = std::ranges::max(data, {}, [=](const auto e){ return e[i]; })[i];

    for (auto& e : data)
      e[i] /= div;
  }

  fmt::print("created data. {} real and {} simulated events\n", real_count, data.size() - real_count);

  return std::pair(data, static_cast<double>(real_count) / static_cast<double>(data.size()));
}

constexpr double logistic(const double val)
{
  return 1.0 / (1.0 + std::exp(-val)) - 0.5;
}

constexpr double derivative_logistic(const double val)
{
  const double exp = std::exp(val);

  return exp / pow<2>(1.0 + exp);
}

constexpr double derivative_logistic_from_logistic(const double val)
{
  return (0.5 + val) * (0.5 - val);
}

int main()
{
  const auto [data, fraction_background] = create_data();

  const auto fraction_signal = 1.0 - fraction_background;

  constexpr std::size_t depth{5};

  [[maybe_unused]] constexpr double dropout{0.5};

  auto connections = []
  {
    std::array<std::array<std::array<double, variable_count>, variable_count>, depth - 1> out;

    for (auto& i : out)
      for (auto& j : i)
        for (auto& k : j)
          k = 1.0 / variable_count * movency::random::fast(movency::random::uniform_distribution(-2.0, 2.0));

    return out;
  }();

  bool train{true};

  std::vector<decltype(connections)> updates(thread_count);

  constexpr std::size_t bucket_count{50};

  std::vector<std::array<std::array<double, 2>, bucket_count>> histograms(thread_count);

  std::atomic<std::ptrdiff_t> reps{0};

  std::ptrdiff_t excess_reps{0};

  auto combine_updates = [&]
  {
    for (auto& update : updates)
      for (std::size_t i = 0; i < depth - 1; ++i)
        for (std::size_t j = 0; j < variable_count; ++j)
          for (std::size_t k = 0; k < variable_count; ++k)
            connections[i][j][k] += update[i][j][k];

    if (!excess_reps)
    {
      for (std::size_t layer = 0; layer < depth - 1; ++layer)
      {
        fmt::print("\nconnections layer {}:\n", layer);

        for (std::size_t i = 0; i < variable_count; ++i)
        {
          for (std::size_t j = 0; j < variable_count; ++j)
            fmt::print("{: f}  ", connections[layer][i][j]);

          fmt::print("\n");
        }
      }

      std::array<std::array<double, 2>, bucket_count> main_histogram{};

      for (auto& histogram : histograms)
        for (std::size_t bucket_no = 0; bucket_no < bucket_count; ++bucket_no)
          for (std::size_t target = 0; target <= 1; ++target)
          {
            main_histogram[bucket_no][target] += histogram[bucket_no][target];

            histogram[bucket_no][target] = 0;
          }

      const auto max_bucket = [&]
      {
        const auto temp = std::ranges::max(main_histogram, {}, [](auto b){ return b[0] + b[1]; });

        return temp[0] + temp[1];
      }();

      if (max_bucket)
      {
        for (const auto& bucket : main_histogram)
        {
          int i = 0;

          for (; i < bucket[0] * 100 / max_bucket; i++)
            fmt::print("0");

          for (; i < (bucket[0] + bucket[1]) * 100 / max_bucket; i++)
            fmt::print("1");

          fmt::print(";\n");
        }

        for (auto& b : main_histogram)
          for (auto& t : b)
            t = 0;
      }

      {
        char input;

        fmt::print("For next iterations, tEst or tRain? ");

        while (true)
        {
          std::cin >> input;

          if (input == 'E')
            train = false;
          else if (input == 'R')
            train = true;
          else
          {
            fmt::print("Invalid. Try again. ");
            continue;
          }

          break;
        }
      }

      fmt::print("\nInput rep count: ");
      std::cin >> excess_reps;
    }

    reps = std::min(excess_reps, 100l);

    excess_reps -= reps;
    /*fmt::print("ers: {}\n", excess_reps);

    for (auto r : reps)
      fmt::print(" {}", r);

    fmt::print("\n");*/

    return;
  };

  std::barrier sync_point(static_cast<std::ptrdiff_t>(thread_count), combine_updates);

  //bool print_next = true;

  std::vector<std::mutex> update_mutexes(thread_count);

  const std::size_t train_cutoff_index = (data.size() * 9) / 10;

  auto run_iteration = [&](const auto thread_no)
  {
    while (true)
    {
      //fmt::print("\none\n");
      while (true)
      {
        const auto r = reps.fetch_sub(1, std::memory_order_relaxed);

        if (r <= 0)
          break;

        //fmt::print("\ntwo\n");
        auto index = [&]() -> std::size_t
        {
          if (train)
            return movency::random::fast(movency::random::uniform_distribution(std::size_t{0}, train_cutoff_index - 1));
          else 
            return movency::random::fast(movency::random::uniform_distribution(train_cutoff_index, data.size() - 1));
        }();

        std::array<std::array<double, variable_count>, depth> nodes;

        for (std::size_t i = 0; i < variable_count; ++i)
        {
          nodes[0][i] = data[index][i];
          //fmt::print("*{} ", nodes[0][i]);
        }
        //fmt::print("{}\n\n", data[index][full_variable_count]);

        for (std::size_t i = 1; i < depth; ++i)
          for (std::size_t j = 0; j < variable_count; ++j)
          {
            nodes[i][j] = 0;

            for (std::size_t k = 0; k < variable_count; ++k)
            {
              nodes[i][j] += nodes[i - 1][k] * connections[i - 1][j][k];
            }

            nodes[i][j] = logistic(nodes[i][j]);
            //fmt::print("***{} ", nodes[i][j]);
          }

        double score = 0;

        for (std::size_t i = 0; i < variable_count; ++i)
          score += nodes.back()[i];

        score = logistic(score) + 0.5;

        // begin backpropagation of errors

        const auto target = data[index][full_variable_count];

        const auto bias = target ? fraction_background : fraction_signal;

        histograms[thread_no][static_cast<std::size_t>(score * bucket_count)][static_cast<std::size_t>(target)] += bias;
        //fmt::print("s{}, stb{}\n", score, score * bucket_count);

        const auto error = (target - score) * bias * derivative_logistic_from_logistic(score - 0.5);

        decltype(nodes) errors;

        for (std::size_t i = 0; i < variable_count; ++i)
          errors.back()[i] = error / variable_count * derivative_logistic_from_logistic(nodes.back()[i]);

        for (std::size_t i = depth - 2; i > 0; --i)
          for (std::size_t j = 0; j < variable_count; ++j)
          {
            errors[i][j] = 0;

            for (std::size_t k = 0; k < variable_count; ++k)
              errors[i][j] += errors[i + 1][k] * connections[i][k][j];

            errors[i][j] *= derivative_logistic_from_logistic(nodes[i][j]);
            //errors[i][j] *= 4*derivative_logistic_from_logistic(nodes[i][j]);
          }

        if (train)
        {
          // begin updating of weights

          for (std::size_t i = 0; i < depth - 1; ++i)
            for (std::size_t j = 0; j < variable_count; ++j)
              for (std::size_t k = 0; k < variable_count; ++k)
                updates[thread_no][i][j][k] += errors[i + 1][j] * nodes[i][k];
        }

        //if (movency::random::fast(movency::random::uniform_distribution<bool>(0.00001)))
        //if (movency::random::fast(movency::random::uniform_distribution<bool>(0.001)))
        /*if (false)
        {
          if (data[index][full_variable_count] == print_next)
          {
            //fmt::print("                                                       ");
            if (print_next)
              fmt::print("\n");
            else
              fmt::print("       ");

            fmt::print("score: {:>.7f}, {:>.7f}", score, target);

            print_next = !print_next;
          }
        }*/

        //if (movency::random::fast(movency::random::uniform_distribution<bool>(0.00001)))
        if(false)
        //if (!reps[thread_count]--)
        {
          fmt::print("\n\nscore: {}, {}\nnodes:\n", score, target);

          for (std::size_t j = 0; j < variable_count; ++j)
          {
            for (std::size_t i = 0; i < depth; ++i)
              fmt::print("{: f}  ", nodes[i][j]);

            fmt::print("\n");
          }

          fmt::print("\n\nerrors:\n");

          for (std::size_t j = 0; j < variable_count; ++j)
          {
            fmt::print(" unused    ");

            for (std::size_t i = 1; i < depth; ++i)
              fmt::print("{: f}  ", errors[i][j]);

            fmt::print("\n");
          }

          fmt::print("\n");

          /*for (std::size_t layer = 0; layer < depth - 1; ++layer)
          {
            fmt::print("\nconnections layer {}:\n", layer);

            for (std::size_t i = 0; i < variable_count; ++i)
            {
              for (std::size_t j = 0; j < variable_count; ++j)
                fmt::print("{: f}  ", connections[layer][i][j]);

              fmt::print("\n");
            }
          }*/
        }
      }
      sync_point.arrive_and_wait();

      updates[thread_no] = {};
    }
  };

  {
    std::vector<std::jthread> loop_threads{};

    for (std::uint32_t i = 0; i < thread_count; ++i)
      loop_threads.emplace_back(run_iteration, i);

    fmt::print("Spawned {} threads.\n", thread_count);
  }

  fmt::print("done\n");

  return EXIT_SUCCESS;
}

