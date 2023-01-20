#include "root.hpp"

#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TLatex.h"
#include "TAxis.h"

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
#include <functional>


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


const std::size_t thread_count{std::max(1u, std::thread::hardware_concurrency() - 1)};
//constexpr std::size_t thread_count{4};

// create a thread for each core and run the given function on each
// the function must take 0 arguments, or a single std::uint32_t,
// in which case the thread number [0, thread_count - 1]  will be passed
void do_threaded_without_pool(auto func)
{
  std::vector<std::jthread> loop_threads{};

  loop_threads.reserve(thread_count);

  for (std::uint32_t thread_index = 0; thread_index < thread_count; ++thread_index)
    if constexpr (std::invocable<decltype(func)>)
      loop_threads.emplace_back(func);
    else
    {
      static_assert(std::invocable<decltype(func), std::uint32_t>, "test_repeat must be passed a function callable with 0 or 1 arguments");

      loop_threads.emplace_back(func, thread_index);
    }
}


std::vector<std::function<void()>> thread_functions(thread_count);

std::array thread_blockers = 
  ( []<std::size_t... I>(std::index_sequence<I...>)
    { return std::array<std::binary_semaphore, sizeof...(I)>{{ ((void) I, std::binary_semaphore(0))... }}; }
  )(std::make_index_sequence<100>{});

std::array thread_completers = 
  ( []<std::size_t... I>(std::index_sequence<I...>)
    { return std::array<std::binary_semaphore, sizeof...(I)>{{ ((void) I, std::binary_semaphore(0))... }}; }
  )(std::make_index_sequence<100>{});

void thread_loop(const std::uint32_t thread_index)
{
  while (true)
  {
    thread_blockers[thread_index].acquire();

    thread_functions[thread_index]();

    thread_completers[thread_index].release();
  }
}

std::vector<std::thread> threads = []
{
  std::vector<std::thread> out;

  out.reserve(thread_count);

  for (std::size_t thread_index = 0; thread_index < thread_count; ++thread_index)
    out.emplace_back(thread_loop, thread_index);

  return out;
}();

// run the given function on each thread in a thread pool
// the function must take 0 arguments, or a single std::uint32_t,
// in which case the thread number [0, thread_count - 1]  will be passed
void do_threaded(auto func, const std::size_t thread_max = thread_count)
{
  for (std::uint32_t thread_index = 0; thread_index < thread_max; ++thread_index)
  {
    if constexpr (std::invocable<decltype(func)>)
      thread_functions[thread_index] = func;
    else
    {
      static_assert(std::invocable<decltype(func), std::uint32_t>, "test_repeat must be passed a function callable with 0 or 1 arguments");

      thread_functions[thread_index] = std::bind(func, thread_index);
    }

    thread_blockers[thread_index].release();
  }

  for (std::uint32_t thread_index = 0; thread_index < thread_max; ++thread_index)
    thread_completers[thread_index].acquire();
}

// call the given function on all values in the range [0, n - 1]
// uses the thread pool
// provides no guarantees on execution order or which thread functions are executed on
void loop_threaded(auto func, const std::size_t n)
{
  static_assert(std::invocable<decltype(func), std::size_t>, "test_repeat must be passed a function callable with 1 arguments");

  std::atomic<std::size_t> global_index{0};

  auto do_next_index = [&]()
  {
    while (true)
    {
      const auto index = global_index.fetch_add(1, std::memory_order_relaxed);

      if (index >= n)
        break;

      func(index);
    }
  };

  if (n < thread_count)
    do_threaded(do_next_index, n);
  else
    do_threaded(do_next_index);
}


using namespace std::literals;

constexpr auto variable_names = std::to_array<std::string_view>({"Lres_IPCHI2_OWNPV"sv, "h1_P"sv, "h1_PT"sv, "h2_P"sv, "h2_PT"sv, "Lres_FD_OWNPV"sv, "Jpsi_FD_OWNPV"sv, "Lres_TAUCHI2"sv, "Lb_IP_OWNPV"sv, "Jpsi_P"sv, "Jpsi_ENDVERTEX_CHI2"sv, "Lres_ENDVERTEX_CHI2"sv});

constexpr std::size_t variable_count{variable_names.size()};

namespace filtering_variables // things involving variables not used in training are contained here
{
  constexpr auto cut_only_variable_names = std::to_array<std::string_view>({"Lb_M", "Lb_BKGCAT", "Jpsi_M"});

  constexpr auto all_variable_names = []
  {
    std::array<std::string_view, variable_count + cut_only_variable_names.size()> out;

    std::size_t i{0};

    for (; i < variable_count; ++i)
      out[i] = variable_names[i];

    for (; i < out.size(); ++i)
      out[i] = cut_only_variable_names[i - variable_count];

    return out;
  }();

  constexpr std::size_t full_variable_count{all_variable_names.size()};

  // convert a variable name into an index into all_variable_names at compile time
  consteval auto index (const std::string_view name)
  {
    auto it = std::ranges::find(all_variable_names, name);

    if (it == all_variable_names.end())
      throw "variable name not in all_variable_names";

    return static_cast<std::size_t>(std::distance(all_variable_names.begin(), it));
  };
}

//TODO: filter data before all has been read to avoid excessive memory usage????
auto create_data()
{
  using namespace filtering_variables;

  std::vector<std::array<double, full_variable_count + 1>> data{};

  // reads desired variables from a file, appending them to data, marking them with the given score (ie background or signal)
  auto read_from_file = [&](const movency::root::file& file, const bool score)
  {
    fmt::print("reading from file {}\n", file.get_path());

    auto old_size = data.size();

    std::mutex resize_lock;

    // reads in a single variable from the file to data, thread-safely enlarging data as necessary
    auto read_variable = [&](std::size_t variable_index)
    {
      const auto variables = [&]
      {
        if (variable_index != index("Lb_BKGCAT"))
          return file.uncompress<double>(std::string(all_variable_names[variable_index]).c_str());

        // Lb_BKGCAT is only present in sim files, and is not stored as doubles
        if (score)
        {
          const auto temp = file.uncompress<int>(std::string(all_variable_names[variable_index]).c_str());

          return std::vector<double>(temp.begin(), temp.end());
        }
        else
          return std::vector<double>(file.get_size<double>(std::string(all_variable_names[0]).c_str()));
      }();

      {
        std::scoped_lock lock(resize_lock);

        if (data.size() == old_size)
          data.resize(old_size + variables.size());
      }

      if (old_size + variables.size() != data.size())
      {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: Inconsistent variable counts in file {}. Previous count: {}. Variable count: {}. Expected total: {}. Variable: {}\n", file.get_path(), old_size, variables.size(), data.size(), all_variable_names[variable_index]);

        std::exit(EXIT_FAILURE);
      }

      fmt::print("read {} {} {} values\n", variables.size(), score ? "simulated" : "real", all_variable_names[variable_index]);

      for (std::size_t i = 0; i < variables.size(); ++i)
      {
        data[old_size + i][variable_index] = variables[i];
      }
    };

    loop_threaded(read_variable, full_variable_count);

    for (std::size_t i = old_size; i < data.size(); ++i)
      data[i][full_variable_count] = score;

    fmt::print("\n");
  };

  /*
  read_from_file({    "../data/Lb2pKmm_mgUp_2016.root"}, 0);
  read_from_file({"../data/Lb2pKmm_sim_mgUp_2016.root"}, 1);
  read_from_file({    "../data/Lb2pKmm_mgDn_2016.root"}, 0);
  read_from_file({"../data/Lb2pKmm_sim_mgDn_2016.root"}, 1);
  read_from_file({    "../data/Lb2pKmm_mgUp_2017.root"}, 0);
  read_from_file({"../data/Lb2pKmm_sim_mgUp_2017.root"}, 1);
  read_from_file({    "../data/Lb2pKmm_mgDn_2017.root"}, 0);
  read_from_file({"../data/Lb2pKmm_sim_mgDn_2017.root"}, 1);
  */
  read_from_file({    "../data/Lb2pKmm_mgUp_2018.root"}, 0);
  read_from_file({"../data/Lb2pKmm_sim_mgUp_2018.root"}, 1);
  read_from_file({    "../data/Lb2pKmm_mgDn_2018.root"}, 0);
  read_from_file({"../data/Lb2pKmm_sim_mgDn_2018.root"}, 1);

  std::erase_if(data, [](const auto e){ return (std::abs(e[index("Lb_M")] - 5619.60) < 300.0 ) != e[full_variable_count]; });

  std::erase_if(data, [](const auto e)
    { 
      const double Jpsi_M = e[index("Jpsi_M")];

      const double q2 = pow<2>(Jpsi_M)/1000000.0;

      return (q2 > 0.98 && q2 < 1.10) || (q2 > 8.0 && q2 < 11.0) || (q2 > 12.5 && q2 < 15.0);
    });

  std::erase_if(data, [](const auto e){ return e[index("Lb_BKGCAT")] != 0 && e[index("Lb_BKGCAT")] != 10 && e[index("Lb_BKGCAT")] != 50; });

  const auto real_count = static_cast<std::size_t>(std::ranges::count_if(data, [](const auto e){ return !e[full_variable_count]; }));

  std::vector<std::array<double, variable_count + 1>> out{data.size()};

  // copies a variable from data to out, and normalizes it
  auto normalize_variable = [&](std::size_t variable_index)
  {
    if (variable_index == variable_count) // this is the score variable, so don't need to normalize and has special index
    {
      for (std::size_t i = 0; i < data.size(); ++i)
        out[i][variable_count] = data[i][full_variable_count];

      return;
    }

    const double div = std::ranges::max(data, {}, [=](const auto e){ return std::abs(e[variable_index]); })[variable_index];

    for (std::size_t i = 0; i < data.size(); ++i)
      out[i][variable_index] = data[i][variable_index] / div;
  };

  loop_threaded(normalize_variable, variable_count + 1);

  std::ranges::shuffle(out, movency::random::prng_);

  fmt::print("created data. {} real and {} simulated events\n", real_count, data.size() - real_count);

  return std::pair(out, static_cast<double>(real_count) / static_cast<double>(out.size()));
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


auto canvas = std::make_unique<TCanvas>("canvas", "canvas", 3000, 1900); //make before creation of r to avert root segfault (magic!)


// creates histograms and saves them to a file
template<floating T>
void create_histogram(const std::span<const T> predictions, const std::string name, const std::span<const std::array<T, variable_count + 1>> data, const std::size_t train_cutoff_index)
{
  if (predictions.size() != data.size())
  {
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "sizes of predictions ({}) and data ({}) not equal\n", predictions.size(), data.size());

    std::exit(1);
  }

  TMultiGraph mgraph{name.c_str(), (name+";score;log10(count)").c_str()};

  const std::array predictions_subs{predictions.subspan(0, train_cutoff_index), predictions.subspan(train_cutoff_index)};
  const std::array        data_subs{       data.subspan(0, train_cutoff_index),        data.subspan(train_cutoff_index)};

  for (std::size_t test = 0; test < 2; ++test)
  {
    const double fraction_background = static_cast<double>(std::ranges::count_if(data_subs[test], [](const auto e){ return !e[variable_count]; })) / static_cast<double>(data_subs[test].size());

    constexpr std::size_t bucket_count{1000};

    std::array<std::array<double, 2>, bucket_count> histogram{};

    // accumulate points into histogram buckets
    for (std::size_t i = 0; i < predictions_subs[test].size(); ++i)
    {
      const auto target = data_subs[test][i][variable_count];

      const auto bias = target ? fraction_background : 1 - fraction_background;

      histogram[static_cast<std::size_t>(predictions_subs[test][i] * bucket_count)][static_cast<std::size_t>(target)] += bias;
    }

    auto signal_graph{std::make_unique<TGraph>()};
    auto backgr_graph{std::make_unique<TGraph>()};

    if (test)
    {
      signal_graph->SetLineColor(kBlue);
      backgr_graph->SetLineColor(kRed);
    }

    for (std::size_t i = 0; i < histogram.size(); ++i)
    {
      if (histogram[i][1])
        signal_graph->AddPoint(static_cast<double>(i)     / static_cast<double>(histogram.size()),
                               std::log10(histogram[i][1] / static_cast<double>(data_subs[test].size())));

      if (histogram[i][0])
        backgr_graph->AddPoint(static_cast<double>(i)     / static_cast<double>(histogram.size()),
                               std::log10(histogram[i][0] / static_cast<double>(data_subs[test].size())));
    }

    mgraph.Add(signal_graph.release());
    mgraph.Add(backgr_graph.release());
  }

  mgraph.Draw("a");

  canvas->SaveAs(fmt::format("cache/{}.png", name).c_str());
}

// creates ROC curves and saves them to a file, along with AUC values
template<floating T>
void create_ROC_curve(const std::span<const T> predictions, const std::string name, const std::span<const std::array<T, variable_count + 1>> data, const std::size_t train_cutoff_index)
{
  if (predictions.size() != data.size())
  {
    fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "sizes of predictions ({}) and data ({}) not equal\n", predictions.size(), data.size());

    std::exit(1);
  }

  const std::array predictions_subs{predictions.subspan(0, train_cutoff_index), predictions.subspan(train_cutoff_index)};
  const std::array        data_subs{       data.subspan(0, train_cutoff_index),        data.subspan(train_cutoff_index)};

  // create and fill 4 vectors with background and signal predictions from the testing and training datasets
  std::array<std::vector<T>, 2> background_predictions;
  std::array<std::vector<T>, 2> signal_predictions;

  for (std::size_t test = 0; test < 2; ++test)
    for (std::size_t i = 0; i < data_subs[test].size(); ++i)
      if (data_subs[test][i][variable_count])
        signal_predictions[test].emplace_back(predictions_subs[test][i]);
      else
        background_predictions[test].emplace_back(predictions_subs[test][i]);

  std::array<double, 2> area{0, 0};

  TMultiGraph mgraph{name.c_str(), (name + ";False positive rate;True positive rate").c_str()};

  {
    const std::array orig_signal_count    {static_cast<double>(    signal_predictions[0].size()), static_cast<double>(    signal_predictions[1].size())};
    const std::array orig_background_count{static_cast<double>(background_predictions[0].size()), static_cast<double>(background_predictions[1].size())};

    constexpr std::size_t point_count{10000};

    std::array<double, 2> prev_signal    {1.0, 1.0};
    std::array<double, 2> prev_background{1.0, 1.0};

    std::array graph{std::make_unique<TGraph>(), std::make_unique<TGraph>()};

    for (std::size_t i = 1; i < point_count + 1; ++i)
    {
      const double cut = static_cast<double>(i) / point_count;

      auto cutdown = [&](std::size_t j)
      {
        if (j / 2)
          std::erase_if(signal_predictions    [j % 2], [=](T v){ return v < cut; });
        else
          std::erase_if(background_predictions[j % 2], [=](T v){ return v < cut; });
      };

      loop_threaded(cutdown, 4);

      const std::array next_signal    {static_cast<double>(    signal_predictions[0].size()) /     orig_signal_count[0],
                                       static_cast<double>(    signal_predictions[1].size()) /     orig_signal_count[1]};
      const std::array next_background{static_cast<double>(background_predictions[0].size()) / orig_background_count[0],
                                       static_cast<double>(background_predictions[1].size()) / orig_background_count[1]};

      for (std::size_t test = 0; test < 2; ++test)
      {
        area[test] += (prev_background[test] - next_background[test]) * (next_signal[test] + prev_signal[test]) / 2;

        prev_signal    [test]  = next_signal    [test];
        prev_background[test]  = next_background[test];
      }

      graph[0]->SetLineColor(kGray);

      graph[0]->AddPoint(next_background[0], next_signal[0]);
      graph[1]->AddPoint(next_background[1], next_signal[1]);
    }

    mgraph.Add(graph[0].release());
    mgraph.Add(graph[1].release());
  }

  std::string label = fmt::format("#splitline{{AUC = {} (test)}}{{AUC = {} (train)}}", area[0], area[1]);

  {
    auto latex = std::make_unique<TLatex>(0.25, 0.5, label.c_str());
    mgraph.GetListOfFunctions()->Add(latex.release());
  }

  mgraph.Draw("a");

  canvas->SaveAs(fmt::format("cache/{}.png", name).c_str());
}

// prints a 3D network of connections
template<class T>
void print_connections(const T& connections)
{
  static_assert(floating<typename T::value_type::value_type::value_type>, "connections must be a 3D array of floating point type values");

  static_assert(std::tuple_size_v<typename T::value_type> == std::tuple_size_v<typename T::value_type::value_type>, "connection layers must be square");

  for (std::size_t layer = 0; layer < connections.size(); ++layer)
  {
    fmt::print("\nconnections layer {}:\n", layer);

    for (const auto& weights : connections[layer])
    {
      for (const auto weight : weights)
        fmt::print("{: f}  ", weight);

      fmt::print("\n");
    }
  }
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

  const std::size_t train_cutoff_index = (data.size() * 9) / 10;

  std::vector<double> predictions(data.size());

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
      if (!train)
      {
        create_histogram(std::span<const double>(predictions), "log_predictions", {data}, train_cutoff_index);

        create_ROC_curve(std::span<const double>(predictions), "ROC_curve",       {data}, train_cutoff_index);
      }

      char input;

      while (true)
      {
        fmt::print("Input: Print current connections, perform a tEst, or tRain for an optional number of iterations? ");

        std::cin >> input;

        if (input == 'P')
        {
          print_connections(connections);

          continue;
        }
        else if (input == 'E')
        {
          train = false;

          reps = 0;
        }
        else if (input == 'R')
        {
          train = true;

          fmt::print("\nInput rep count: ");
          std::cin >> excess_reps;
        }
        else
        {
          fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "Invalid input\n");

          continue;
        }

        break;
      }
    }

    reps = std::min(excess_reps, 100l);

    excess_reps -= reps;

    return;
  };

  std::barrier sync_point(static_cast<std::ptrdiff_t>(thread_count), combine_updates);

  auto iterate = [&](const auto thread_no)
  {
    while (true)
    {
      while (true)
      {
        std::size_t index;

        std::ptrdiff_t r;

        if (train)
        {
          r = reps.fetch_sub(1, std::memory_order_relaxed);

          index = movency::random::fast(movency::random::uniform_distribution(std::size_t{0}, train_cutoff_index - 1));

          if (r <= 0)
            break;
        }
        else
        {
          r = reps.fetch_add(1, std::memory_order_relaxed);

          index = static_cast<std::size_t>(r);

          if (index >= data.size())
            break;
        }

        std::array<std::array<double, variable_count>, depth> nodes;

        for (std::size_t i = 0; i < variable_count; ++i)
        {
          nodes[0][i] = data[index][i];
          //fmt::print("*{} ", nodes[0][i]);
        }
        //fmt::print("{}\n\n", data[index][variable_count]);

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

        if (train)
        {
          // begin backpropagation of errors

          const auto target = data[index][variable_count];

          const auto bias = target ? fraction_background : fraction_signal;

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

          // save updates to make to weights

          for (std::size_t i = 0; i < depth - 1; ++i)
            for (std::size_t j = 0; j < variable_count; ++j)
              for (std::size_t k = 0; k < variable_count; ++k)
                updates[thread_no][i][j][k] += errors[i + 1][j] * nodes[i][k];
        }
        else // record the score given to this event
          predictions[index] = score;
      }

      sync_point.arrive_and_wait();

      updates[thread_no] = {};
    }
  };

  do_threaded_without_pool(iterate);

  fmt::print("done\n");

  return EXIT_SUCCESS;
}

