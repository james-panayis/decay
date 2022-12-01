#include "root.hpp"

#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "TLatex.h"
#include "ROOT/RDataFrame.hxx"

#include "timeblit/random.hpp"

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <memory>
#include <thread>
#include <array>
#include <algorithm>
#include <fstream>
#include <charconv>

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


auto fit()
{
  //ROOT::EnableImplicitMT(); // Breaks saving of graphs (causes seperate (inaccessible???) canvases for each(?) thread)

  // first list has particles of charge 0; second has particles of charge +-1
  const auto lists = []
  {
    std::array<std::vector<particle_info>, 2> out_lists;

    std::ifstream in("../data/ParticleTable.txt");

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

      if (charge && out_lists[charge].size() != 0)
      {
        const auto prev = out_lists[charge].back();
      
        if (   name.substr(0, name.size() - 1) == prev.name.substr(0, prev.name.size() - 1)
            && name.back() != prev.name.back()
            && mass == prev.mass)
          continue; // same particle with opposite charge
      }

      out_lists[charge].emplace_back(name, mass, width);

      //fmt::print("{}\n", out_lists[charge].back());
    }

    return out_lists;
  }();

  fmt::print("\nLists created. Sizes(charge): {}(0) {}(+-1)\n\n", lists[0].size(), lists[1].size());
  

  //ROOT::RDataFrame rdf{"Masses", "cache/mass.root"};
  const movency::root::file r("cache/mass.root");

  //const auto cols = rdf.GetColumnNames();
  const auto cols = r.get_names();

  auto canvas = std::make_unique<TCanvas>("canvas", "canvas", 3000, 1900);

  std::mutex canvas_mutex;

  std::atomic<std::uint32_t> next{0};

  //for (int n = 0; n < std::ssize(vecs); ++n)
  auto loop = [&]
  {
    enum class daughter {z, e, mu, pi, k, p};

    auto get_daughters = [](const std::string_view name)
    {
      std::size_t i = 0;

      std::array<daughter, 4> daughters;

      //for (std::size_t d_num = 0; d_num < 4; ++d_num)
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
      }

      return daughters;
    };

    while (true)
    {
      const auto n = next.fetch_add(1, std::memory_order_relaxed);

      if (n >= cols.size())
        return;

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

      fmt::print("about to read variable: {}\n", cols[n].first);

      const std::vector<double> vec = [&]
      {
        fmt::print("reading variable: {}\n", cols[n].first);

        //std::vector<double> out = *rdf.Take<double>(cols[n].first);
        std::vector<double> out = r.uncompress<double>(cols[n].first);

        std::ranges::sort(out);

        return out;
      }();

      fmt::print("fitting variable: {}\n", cols[n].first);

      if (vec.empty())
      {
        fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: No data present for variable {} (index n={})\n", cols[n].first, n);

        //std::exit(1);
        continue;
      }

      const auto min = vec.front();
      const auto max = vec.back();

      const auto span = max - min;

      constexpr int bucket_count{1000};

      constexpr double spread{2.0}; // how far each point should bleed into neighbouring buckets (linearly)

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

        for (const double val : vec)
        {
          const double distance = (val - min) / span * static_cast<double>(bucket_count);

          const auto from = static_cast<std::uint32_t>(std::max(0,          static_cast<std:: int32_t>(distance - spread)    ));
          const auto to   = static_cast<std::uint32_t>(std::min(out.size(), static_cast<std::uint64_t>(distance + spread) + 1));

          for (std::uint32_t i = from; i < to; ++i)
            out[i] += std::max(0.0, spread - std::abs(dist_vals[i] - min) / span);
        }

        return out;
      }();

      std::vector<peak_t> peaks{};

      for (auto _l = 750; _l--;)
      {
        auto calculate_residuals = [&]<bool ignore = false>(const std::size_t ignore_index = 0)
        {
          std::array<double, bucket_count> out;

          for (std::size_t i = 0; i < out.size(); ++i)
          {
            out[i] = distribution[i];

            for (std::size_t j = 0; j < peaks.size(); ++j)
            {
              if constexpr (ignore)
                if (j == ignore_index)
                  continue;

              const double offset = std::abs(peaks[j].position - dist_vals[i]) / peaks[j].width;

              out[i] -= peaks[j].magnitude * std::exp(-pow<2>(offset));
            }
          }

          return out;
        };

        auto calculate_fit = [&](const std::array<double, bucket_count>& residuals, const peak_t peak = {0,0,0})
        {
          double fit{};

          for (std::uint32_t i = 0; i < bucket_count; ++i)
          {
            const double offset = std::abs(peak.position - dist_vals[i]) / peak.width;

            const double val = peak.magnitude * std::exp(-pow<2>(offset));

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
                    variable /= factor;
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

          if (new_fit < prev_fit)
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

        if (peak.magnitude / peak.width < 50)
          break;

        std::ranges::sort(local_list, std::ranges::less{}, [&](const particle_info p){return std::abs(peak.position - p.mass);});

        bool annotated = false;

        for (const particle_info& particle : local_list)
        {
          if (particle.width != std::numeric_limits<double>::infinity() && particle.width * 0.9 > peak.width * 1.665109)
            continue; // peak too thin given particle width

          if (std::abs(peak.position - particle.mass) > 125)
            break;

          //if (particles_so_far.contains(p.name))
          if (std::ranges::find(particles_so_far, particle.name) != particles_so_far.end())
            continue;

          if (!annotated)
            if (peak.magnitude / peak.width > 175)
              if (std::abs(peak.position - particle.mass) < 75)
              {
                annotations.emplace_back(j, particle.name);

                annotated = true;
              }

          particles_so_far.push_back(particle.name);

          fmt::print("{}  (peak {}) dist {}\n", particle, peak, std::abs(peak.position - particle.mass));
        }
      }

      {
        const std::scoped_lock lock(canvas_mutex);

        TMultiGraph mgraph{};

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

          //TLatex* latex = new TLatex(graph->GetX()[100], graph->GetY()[100], "test");
          //graph->GetListOfFunctions()->Add(latex);

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

        mgraph.SetTitle(std::string(cols[n].first).data());

        mgraph.Draw("a");

        canvas->SaveAs(fmt::format("cache/graph_{}.png", cols[n].first).c_str());
      }

      fmt::print("finished with variable {} ({}/{})\n\n", cols[n].first, n + 1, cols.size());
    }
  };

  {
    std::vector<std::jthread> loop_threads{};

    for (std::uint32_t i = 1; i < std::thread::hardware_concurrency(); ++i)
    //for (std::uint32_t i = 1; i < 2; ++i)
      loop_threads.emplace_back(loop);

    fmt::print("Spawned {} threads.\n", loop_threads.size());
  }
}

int main()
{
  fit();
}

