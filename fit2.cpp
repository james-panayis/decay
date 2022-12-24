#include "root.hpp"

#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TLatex.h"

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



struct peak_t
{
  double position;
  double magnitude;
  double width;
};

template <>
struct fmt::formatter<peak_t>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.end();
  }

  template<class FormatContext>
  auto format(const peak_t& p, FormatContext& ctx)
  {
    return fmt::format_to(ctx.out(), FMT_COMPILE("p:{} m:{} w:{}"), p.position, p.magnitude, p.width);
  }
};



struct particle_info
{
  std::string name;
  double      mass;
  //bool        charge; // true means +-1; false means 0
  //bool        even;   // true means even number of quarks 
  double      width;
};

template <>
struct fmt::formatter<particle_info>
{
  constexpr auto parse(format_parse_context& ctx)
  {
    return ctx.end();
  }

  template<class FormatContext>
  auto format(const particle_info& p, FormatContext& ctx)
  {
    //return fmt::format_to(ctx.out(), FMT_COMPILE("{} {} {}"), p.name, p.charge ? 1 : 0, p.mass);
    //return fmt::format_to(ctx.out(), FMT_COMPILE("{} {}"), p.name, p.mass);
    return fmt::format_to(ctx.out(), FMT_COMPILE("{} {} {}"), p.name, p.mass, p.width);
  }
};


// Read file with list of particle information, and return two lists (for particles with 0 and +-1 charge)
auto generate_particle_lists(const std::string filepath)
{
  std::array<std::vector<particle_info>, 2> lists;

  std::ifstream in(filepath);

  std::string line;

  while (std::getline(in, line))
  {
    if (line.starts_with("#") || line == "PARTICLE" || line == "END PARTICLE")
      continue; //comment or header line

    const std::size_t size{line.size()};

    const std::string name = [&]
    {
      std::size_t i = 1;

      while (i != size && line[i] != ' ')
        ++i;

      return line.substr(1, i - 1);
    }();

    //if (name.contains('~'))
    if (std::ranges::find(name, '~') != name.end())
      continue; // antiparticle

    const std::string charge_string{line.substr(40, 4)};

    if (charge_string[0] != ' ' && charge_string[0] != '-')
      continue;

    if (charge_string[1] != '0' && charge_string[1] != '1')
      continue;

    if (charge_string[2] != '.')
      continue;

    if (charge_string[3] != '0')
      continue;

    const bool charge = charge_string[1] == '0' ? false : true;

    const double mass = [&]
    {
      double out;

      std::size_t i = 45;

      while (i != size && line[i] == ' ')
        ++i;

      std::from_chars(line.data() + i, line.data() + 62, out);

      return out * 1000.0;
    }();

    const double width = [&]
    {
      double out;

      std::size_t i = 61;

      while (i != size && line[i] == ' ')
        ++i;

      std::from_chars(line.data() + i, line.data() + 62, out);

      return 6.582119569 * pow<16>(0.1) / out / 1'000'000.0; //hbar in eVs
    }();

    if (charge && lists[charge].size() != 0)
    {
      const auto prev = lists[charge].back();
    
      if (   name.substr(0, name.size() - 1) == prev.name.substr(0, prev.name.size() - 1)
          && name.back() != prev.name.back()
          && mass == prev.mass)
        continue; // same particle with opposite charge
    }

    lists[charge].emplace_back(name, mass, width);

    //fmt::print("{}\n", lists[charge].back());
  }

  return lists;
}


enum class daughter {z, e, mu, pi, k, p};

// Read variable name and return array of daughters represented by variable
constexpr auto get_daughters(const std::string_view name)
{
  std::size_t i = 0;

  std::array<daughter, 4> daughters;

  for (auto& d : daughters)
  {
    switch (name[i++])
    {
      case '0': d = daughter::z;       break;
      case 'e': d = daughter::e;       break;
      case 'k': d = daughter::k;       break;
      case 'm': d = daughter::mu; ++i; break;
      case 'p':
                if (i != name.size() && name[i] == 'i')
                {
                  d = daughter::pi;
                  ++i;
                }
                else
                  d = daughter::p;
                break;
      default:
                fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: invalid column character {} in column {}\n", i, name);
    }

    ++i;
  }

  return daughters;
};


struct peak_record
{
  double        score;
  std::uint32_t graph_idx;
  std::uint32_t peak_idx;
  std::string   name;
};

  
auto fit()
{
  // first list has particles of charge 0; second has particles of charge +-1
  const auto lists{generate_particle_lists("../data/ParticleTable.txt")};

  fmt::print("\nLists created. Sizes(charge): {}(0) {}(+-1)\n\n", lists[0].size(), lists[1].size());

  const std::size_t thread_count{std::max(1u, std::thread::hardware_concurrency() - 1)};
  //const std::size_t thread_count{1};

  auto canvas = std::make_unique<TCanvas>("canvas", "canvas", 1500, 950); //make before creation of r to avert root segfault (magic!)

  const movency::root::file r("cache/mass.root");

  //fmt::print("read\n");

  const auto cols = r.get_names();

  const auto event_count = r.uncompress<double>(cols.front().first).size();

  std::vector<double> weights(event_count, 1); // Weight to give each event when constructing distribution

  std::vector<std::vector<peak_t>> peak_sets(event_count);

  std::vector<std::vector<peak_record>> best_local_peaks(thread_count); // score, event idx, peak idx, peak name

  std::mutex canvas_mutex;

  std::atomic<std::uint32_t> next{0};

  constexpr int bucket_count{1000}; // buckets in each histogram

  constexpr double spread{2.0}; // how far each point should bleed into neighbouring buckets (linearly)

  int iteration = 0;

  std::ofstream log{"cache/log.txt"};

  auto remove_peak = [&]
  {
    next = 0;

    std::vector<peak_record> best_peaks{};
    
    best_peaks.reserve(25 * thread_count);

    for (auto& vec : best_local_peaks)
      best_peaks.insert(best_peaks.end(), vec.begin(), vec.end());

    std::ranges::sort(best_peaks, [](const auto& lhs, const auto& rhs){ return lhs.score > rhs.score; });

    for (std::uint32_t i = 0; true; ++i)
    {
      if (i > 25)
        fmt::print("WARNING: exceeded 25 potential peaks\n");

      fmt::print("use decay: {} -> {} ? [y/n]\n", best_peaks[i].name, cols[best_peaks[i].graph_idx].first);

      char in;

      std::cin >> in;

      if (in == 'y')
      {
        const peak_t& peak = peak_sets[best_peaks[i].graph_idx][best_peaks[i].peak_idx];

        log << fmt::format("removing decay: {} -> {}   (peak: {})\n", best_peaks[i].name, cols[best_peaks[i].graph_idx].first, peak);
        log.flush();

        fmt::print("PEAK: {}\n", peak);

        const std::vector<double> vec = r.uncompress<double>(cols[best_peaks[i].graph_idx].first);

        const auto min = std::ranges::min(vec);
        const auto max = std::ranges::max(vec);

        const auto span = max - min;

        const auto dist_vals = [&]
        {
          std::array<double, bucket_count> out;

          for (std::uint32_t j = 0; j < out.size(); ++j)
            out[j] = min + static_cast<double>(j) / static_cast<double>(bucket_count - 1) * span;

          return out;
        }();

        // distribution of values
        const auto distribution = [&]
        {
          std::array<double, bucket_count> out{};

          for (std::size_t j = 0; j < event_count; ++j)
          {
            const double distance = (vec[j] - min) / span * static_cast<double>(bucket_count - 1);

            const auto from = static_cast<std::uint32_t>(std::max(0,          static_cast<std:: int32_t>(distance - spread)    ));
            const auto to   = static_cast<std::uint32_t>(std::min(out.size(), static_cast<std::uint64_t>(distance + spread) + 1));

            for (std::uint32_t k = from; k < to; ++k)
              out[k] += weights[j] * std::max(0.0, spread - std::abs(dist_vals[k] - min) / span);
          }

          return out;
        }();

        for (std::uint32_t j = 0; j < event_count; ++j)
        {
          const double distance = (vec[j] - min) / span * static_cast<double>(bucket_count - 1);

          const auto below = static_cast<std::uint32_t>(std::floor(distance));
          const auto above = static_cast<std::uint32_t>(std::ceil (distance));

          if (distance > distribution.size() - 1 || distance < 0)
          {
            fmt::print("ERROR: {} {} {} {}", distance, distribution.size(), above, below);
            int pauspdsiubf;
            std::cin >> pauspdsiubf;
            continue;
          }

          const double background = (distance - below) * distribution[above] + (above - distance) * distribution[below];

          const double signal = [&]
          {
            double out = 0;

            const double offset_1 = std::abs(peak.position - dist_vals[above]) / peak.width;

            out += (distance - below) * peak.magnitude * std::exp(-pow<2>(offset_1));

            const double offset_2 = std::abs(peak.position - dist_vals[below]) / peak.width;

            out += (above - distance) * peak.magnitude * std::exp(-pow<2>(offset_2));

            return out;
          }();

          if (background != 0)
            weights[j] *= (background - signal) / background;
          else
            fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "{} ({}): {}, {} {} {}  {}\n", j, vec[j], weights[j], background, signal, background - signal, (background - signal) / background);

          //if (weights[j] != weights[j])
          //if (background == 0)
            //fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "{}: {}, {} {} {}  {}\n", j, weights[j], background, signal, background - signal, (background - signal) / background);

          if (weights[j] < 0 || weights[j] > 1)
            fmt::print("*** {} {} {} {} {} {}\n", weights[j], vec[j], below, above, background, signal);

          //if (j < 10000)
            //fmt::print("*** {} {} {} {} {} {}\n", weights[j], vec[j], below, above, background, signal);
        }

        break;
      }
    }

    ++iteration;
  };

  std::barrier sync_point(static_cast<std::ptrdiff_t>(thread_count), remove_peak);

  auto loop = [&](const auto thread_no)
  {
    while (true)
    {
      best_local_peaks[thread_no].resize(25);

      for (auto& record : best_local_peaks[thread_no])
        record.score = 0;

      while (true)
      {
        const auto n = next.fetch_add(1, std::memory_order_relaxed);

        if (n >= cols.size())
          break;

        const auto daughters = get_daughters(cols[n].first);

        const int daughter_count = [&]
        {
          int out{0};

          for (const auto d : daughters)
            if (d != daughter::z)
              ++out;

          return out;
        }();

        if (daughter_count <= 1)
        {
          fmt::print("Skipping due to too few particles: {}\n", cols[n].first);

          continue;
        }

        fmt::print("reading variable: {}\n", cols[n].first);

        const std::vector<double> vec = r.uncompress<double>(cols[n].first);

        if (vec.size() != event_count)
        {
          fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: inconsistent number of entries in each record: {} in {}; {} in {}", event_count, 0, vec.size(), 0);

          std::exit(1);
        }

        if (vec.empty())
        {
          fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: No data present for variable {} (index n={})\n", cols[n].first, n);

          std::exit(1);
          //continue;
        }

        fmt::print("fitting variable: {}\n", cols[n].first);

        const auto min = std::ranges::min(vec);
        const auto max = std::ranges::max(vec);

        const auto span = max - min;

        // values represented by buckets
        const auto dist_vals = [&]
        {
          std::array<double, bucket_count> out;

          for (std::uint32_t i = 0; i < out.size(); ++i)
            out[i] = min + static_cast<double>(i) / static_cast<double>(bucket_count - 1) * span;

          return out;
        }();

        // distribution of values
        const auto distribution = [&]
        {
          std::array<double, bucket_count> out{};

          for (std::size_t j = 0; j < event_count; ++j)
          {
            //if (iteration != 0)
              //fmt::print("START\n\n");
            const double distance = (vec[j] - min) / span * static_cast<double>(bucket_count - 1);

            const auto from = static_cast<std::uint32_t>(std::max(0,          static_cast<std:: int32_t>(distance - spread)    ));
            const auto to   = static_cast<std::uint32_t>(std::min(out.size(), static_cast<std::uint64_t>(distance + spread) + 1));

            for (std::uint32_t i = from; i < to; ++i)
            {
              out[i] += weights[j] * std::max(0.0, spread - std::abs(dist_vals[i] - min) / span);
              //if (iteration != 0 && out[i] != out[i])
                //fmt::print("{} {}, v{} w{} dv{} s{} d{}\n", j, i, vec[j], weights[j], dist_vals[i], span, distance);
            }
          }

          return out;
        }();

        std::vector<peak_t>& peaks{peak_sets[n]};

        for (auto _l = 750; _l--;)
        //for (auto _l = 1; _l--;)
        {
          auto calculate_residuals = [&]<bool ignore = false>(const std::size_t ignore_index = 0)
          {
            std::array<double, bucket_count> out;

            //if (iteration != 0)
              //fmt::print("START\n\n");

            for (std::size_t i = 0; i < out.size(); ++i)
            {
              //if (iteration != 0)
                //fmt::print("{}, {} - ", i, distribution[i]);

              out[i] = distribution[i];

              for (std::size_t j = 0; j < peaks.size(); ++j)
              {
                if constexpr (ignore)
                  if (j == ignore_index)
                    continue;

                const double offset = std::abs(peaks[j].position - dist_vals[i]) / peaks[j].width;

                out[i] -= peaks[j].magnitude * std::exp(-pow<2>(offset));
                //if (iteration != 0 && out[i] != out[i])
                  //fmt::print("** {}   {}   {}\n", peaks[j], dist_vals[i], offset);
              }
            }

            return out;
          };

          auto calculate_fit = [&](const std::array<double, bucket_count>& residuals, const peak_t peak = {0,0,0})
          {
            double fit{};

            //if (iteration != 0)
            //fmt::print("START\n\n");

            for (std::uint32_t i = 0; i < bucket_count; ++i)
            {
              const double offset = std::abs(peak.position - dist_vals[i]) / peak.width;

              const double val = peak.magnitude * std::exp(-pow<2>(offset));
            //if (iteration != 0)
              //fmt::print("{} | {} {} {} | {} {}\n", peak, fit, offset, val, dist_vals[i], residuals[i]);

              fit += pow<2>(val - residuals[i]);
            }

            return fit;
          };

          using namespace random;

          //if (std::ssize(peaks) > 0 && (std::ssize(peaks) > 10 || std::bernoulli_distribution(0.9)(prng_)))
          //if (peaks.size() > 1 && std::bernoulli_distribution(1.0 - std::pow(0.4, peaks.size()))(prng_))
          if (peaks.size() > 1 && random::fast(uniform_distribution<bool>(1.0 - std::pow(0.666, peaks.size()))))
          {
            // which peak to alter
            const std::size_t change_index = random::fast(uniform_distribution(0ul, peaks.size() - 1));

            peak_t cpeak = peaks[change_index];

            // difference between distribution and all peaks excluding the one currently under alteration
            const auto residuals{calculate_residuals.template operator()<true>(change_index)};

            double prev_fit = calculate_fit(residuals, cpeak);
            double new_fit  = calculate_fit(residuals);

            auto optimize_variable = [&]<std::uint32_t var>(double factor)
            {
              for (auto _ = 20; _--;)
              {
                const double fit0 = calculate_fit(residuals, cpeak);
                double fit1;
                double fit2;

                double& variable = var == 0 ? cpeak.position : var == 1 ? cpeak.magnitude : cpeak.width;

                if constexpr (var == 0)
                {
                  fit1 = calculate_fit(residuals, {cpeak.position + factor, cpeak.magnitude, cpeak.width});
                  fit2 = calculate_fit(residuals, {cpeak.position + factor + factor, cpeak.magnitude, cpeak.width});
                }
                else if constexpr (var == 1)
                {
                  fit1 = calculate_fit(residuals, {cpeak.position, cpeak.magnitude * factor, cpeak.width});
                  fit2 = calculate_fit(residuals, {cpeak.position, cpeak.magnitude * factor * factor, cpeak.width});
                }
                else if constexpr (var == 2)
                {
                  fit1 = calculate_fit(residuals, {cpeak.position, cpeak.magnitude, cpeak.width * factor});
                  fit2 = calculate_fit(residuals, {cpeak.position, cpeak.magnitude, cpeak.width * factor * factor});
                }
                //fmt::print("fits:  {}   {}   {}\n", fit0, fit1, fit2);
                //fmt::print("{}  :  {}   {}   {}    ({})\n", var, variable, variable * factor, variable * factor * factor, factor);

                if (fit1 < fit0 && fit1 < fit2)
                {
                  if (fit2 < fit0)
                  {
                    if constexpr (var == 0)
                      variable += factor;
                    else
                      variable *= factor;
                  }

                  if constexpr (var == 0)
                    factor *= 0.5;
                  else
                    factor = std::sqrt(factor);
                }
                else
                {
                  if (fit1 > fit2 && fit1 > fit0)
                    break;

                  if (fit0 < fit2)
                  {
                    if constexpr (var == 0)
                      variable -= factor;
                    else
                    {
                      variable /= factor;

                      if constexpr (var == 2)
                        if (variable <= 0.5)
                        {
                          peaks.erase(peaks.begin() + static_cast<std::int64_t>(change_index));
                          return;
                        }
                    }
                  }

                  if constexpr (var == 0)
                    factor *= 1.5;
                  else
                    factor = std::pow(factor, 1.5);
                }
              }

              //peaks[change_index].position = cpeak.position;
              peaks[change_index] = cpeak;

              return;
            };

            if (new_fit <= prev_fit)
            {
              peaks.erase(peaks.begin() + static_cast<std::int64_t>(change_index));
            }
            else if (random::fast(uniform_distribution<bool>(0.25)))
            {
              optimize_variable.template operator()<0>(random::fast(uniform_distribution(-span/500, span/500)));
            }
            else if (random::fast<bool>())
            {
              optimize_variable.template operator()<1>(random::fast(uniform_distribution(0.5, 1.5)));
            }
            else
            {
              optimize_variable.template operator()<2>(random::fast(uniform_distribution(0.5, 1.5)));
            }
          }
          else
          {
            /*
            peaks.emplace_back(random::fast(uniform_distribution(min, max)), 
                                   random::fast(uniform_distribution(0.0, static_cast<double>(vec.size()) * pow<2>(spread) / bucket_count)),
                                   random::fast(uniform_distribution(2.5, span)));
                                   */

            const auto residuals{calculate_residuals()};

            const auto max_idx = static_cast<std::uint32_t>(std::max_element(residuals.begin(), residuals.end()) - residuals.begin());

            const double position = dist_vals[max_idx];

            const double magnitude = residuals[max_idx];

            double best_fit = std::numeric_limits<double>::infinity();

            double best_width = random::fast(uniform_distribution(2.5, span));

            for (auto _ = 500; _--;)
            {
              const double width = best_width * random::fast(uniform_distribution(9.0/10.0, 10.0/9.0));

              double fit{calculate_fit(residuals, {position, magnitude, width})};

              if (fit < best_fit)
              {
                best_fit = fit;

                best_width = width;
              }
            }

            if (best_fit == std::numeric_limits<double>::infinity())
              //fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: no change in best_fit when attempting to add a new curve\n");
            {}
            else
              peaks.emplace_back(position, magnitude, best_width);
          }
        }

        fmt::print("finished fitting gaussians to {}, with {} peaks:\n", cols[n].first,  peaks.size());

        //for (const peak_t& peak : peaks)
          //fmt::print("{}\n", peak);

        bool charge = daughter_count % 2;

        std::ranges::sort(peaks, std::ranges::greater{}, [](const peak_t peak){return peak.magnitude / peak.width;} );

        auto local_list = lists[charge];

        std::vector<std::string> particles_so_far{};

        // list of peak indexes requiring annotation, and their names
        std::vector<std::pair<std::uint32_t, std::string>> annotations{};

        for (std::uint32_t j = 0; j < peaks.size(); ++j)
        {
          const peak_t& peak = peaks[j];

          const auto sharpness = peak.magnitude / peak.width;

          if (sharpness < 50)
            break;

          std::ranges::sort(local_list, std::ranges::less{}, [&](const particle_info p){return std::abs(peak.position - p.mass);});

          bool annotated = false;

          for (const particle_info& particle : local_list)
          {
            if (particle.width != std::numeric_limits<double>::infinity() && particle.width * 0.9 > peak.width * 1.665109)
              continue; // peak too thin given particle width

            const auto distance = std::abs(peak.position - particle.mass);

            if (distance > 125)
              break;

            //if (particles_so_far.contains(p.name))
            if (std::ranges::find(particles_so_far, particle.name) != particles_so_far.end())
              continue;

            if (!annotated)
              if (sharpness > 125)
                if (distance < 60)
                {
                  annotations.emplace_back(j, particle.name);

                  annotated = true;
                }

            particles_so_far.push_back(particle.name);

            //fmt::print("{}  (peak {}) dist {}\n", particle, peak, std::abs(peak.position - particle.mass));

            const double score = std::log(sharpness) / (distance + 1);

            if (std::isfinite(score) && score > best_local_peaks[thread_no].back().score)
              best_local_peaks[thread_no].back() = peak_record{score, n, j, particle.name};

            std::ranges::sort(best_local_peaks[thread_no], [](const auto& lhs, const auto& rhs){ return lhs.score > rhs.score; });
            //fmt::print("*** {}  {}\n", best_local_peaks[thread_no].front().score, best_local_peaks[thread_no].back().score);
          }
        }

        const std::string name{fmt::format("{}_{}", cols[n].first, iteration)};

        {
          const std::scoped_lock lock(canvas_mutex);

          TMultiGraph mgraph{name.c_str(), name.c_str()};

          // graph of distribution
          {
            auto graph{std::make_unique<TGraph>()};

            graph->SetLineColor(kRed);
            graph->SetLineWidth(2);

            for (std::uint32_t i = 0; i < distribution.size(); ++i)
              graph->AddPoint(dist_vals[i], distribution[i]);

            mgraph.Add(graph.release());
          }

          // graph of fit
          {
            auto graph{std::make_unique<TGraph>()};

            graph->SetLineColor(kBlue);
            graph->SetLineWidth(2);

            for (std::uint32_t i = 0; i < distribution.size(); ++i)
            {
              double val{};

              for (const peak_t& peak : peaks)
              {
                const double offset = std::abs(peak.position - dist_vals[i]) / peak.width;

                val += peak.magnitude * std::exp(-pow<2>(offset));
              }

              graph->AddPoint(dist_vals[i], val);
            }

            mgraph.Add(graph.release());
          }

          // graph of underlying gaussians
          for (const peak_t& peak : peaks)
          {
            auto graph{std::make_unique<TGraph>()};

            graph->SetLineColor(kBlack);
            graph->SetLineWidth(1);

            for (std::uint32_t i = 0; i < distribution.size(); ++i)
            {
              const double offset = std::abs(peak.position - dist_vals[i]) / peak.width;

              graph->AddPoint(dist_vals[i], peak.magnitude * std::exp(-pow<2>(offset)));
            }

            mgraph.Add(graph.release());
          }

          // graph marking centers of underlying gaussians (some with annotation labels)
          for (std::uint32_t i = 0; i < peaks.size(); ++i)
          {
            const peak_t& peak = peaks[i];

            auto graph{std::make_unique<TGraph>()};

            graph->SetLineColor(kBlack);
            graph->SetLineWidth(4);

            graph->AddPoint(peak.position, 0);
            graph->AddPoint(peak.position, peak.magnitude);

            for (auto annotation : annotations)
            {
              if (annotation.first == i)
              {
                auto latex = std::make_unique<TLatex>(graph->GetX()[1], graph->GetY()[1], annotation.second.c_str());
                graph->GetListOfFunctions()->Add(latex.release());
              }
            }

            mgraph.Add(graph.release());
          }

          //mgraph.SetTitle(std::string(cols[n].first).data());
          //mgraph.SetName (std::string(cols[n].first).data());

          mgraph.Draw("a");

          canvas->SaveAs(fmt::format("cache/graph_{}.bmp", name).c_str());
        }

        if (std::system(fmt::format("convert cache/graph_{}.bmp cache/graph{}.png", name, name).c_str()))
        {
          fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "Conversion from bmp to png failed\n");

          std::exit(1);
        }

        if (std::system(fmt::format("rm cache/graph_{}.bmp", name).c_str()))
        {
          fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "Deletion of bmp file failed\n");

          std::exit(1);
        }

        fmt::print("finished with variable {} ({}/{})\n\n", cols[n].first, n + 1, cols.size());
      }

      //if (thread_no == 0)
      sync_point.arrive_and_wait();
      //else
      //{
      //sync_point.arrive_and_drop();
      //return;
      //}
    }
  };

  {
    std::vector<std::jthread> loop_threads{};

    for (std::uint32_t i = 0; i < thread_count; ++i)
      loop_threads.emplace_back(loop, i);

    fmt::print("Spawned {} threads.\n", thread_count);
  }
}

int main()
{
  fit();
}

