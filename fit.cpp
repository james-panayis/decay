#include "TCanvas.h"
#include "TGraph.h"
#include "TMultiGraph.h"
#include "ROOT/RDataFrame.hxx"

#include "timeblit/random.hpp"

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/color.h>

#include <memory>
#include <thread>
#include <array>
#include <algorithm>
#include <random>

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




auto fit()
{
  //ROOT::EnableImplicitMT(); // Breaks saving of graphs (causes seperate (inaccessible???) canvases for each(?) thread)


  //std::ifstream in;
  

  ROOT::RDataFrame rdf{"Masses", "cache/mass.root"};

  const auto cols = rdf.GetColumnNames();

  std::mutex data_mutex;

  auto canvas = std::make_unique<TCanvas>("canvas", "canvas", 3000, 1900);

  std::mutex canvas_mutex;

  std::atomic<std::uint32_t> next{0};

  //for (int n = 0; n < std::ssize(vecs); ++n)
  auto loop = [&]
  {
    while (true)
    {
      auto n = next.fetch_add(1, std::memory_order_relaxed);

      if (n >= cols.size())
        return;

      fmt::print("about to read variable: {}\n", cols[n]);

      const std::vector<double> vec = [&]
      {
        const std::scoped_lock lock(data_mutex);

        fmt::print("reading variable: {}\n", cols[n]);

        std::vector<double> out = *rdf.Take<double>(cols[n]);

        std::ranges::sort(out);

        return out;
      }();

      fmt::print("fitting variable: {}\n", cols[n]);

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

      auto generate_new_peaks = [&]
      {
        using namespace random;

        auto new_peaks = peaks;

        //if (std::ssize(new_peaks) > 0 && (std::ssize(new_peaks) > 10 || std::bernoulli_distribution(0.9)(prng_)))
        //if (new_peaks.size() > 1 && std::bernoulli_distribution(1.0 - std::pow(0.4, new_peaks.size()))(prng_))
        if (new_peaks.size() > 1 && random::fast(uniform_distribution<bool>(1.0 - std::pow(0.4, new_peaks.size()))))
        {
          // index of gaussian to alter
          const std::size_t change_index = random::fast(uniform_distribution(0ul, new_peaks.size() - 1));

          //if (std::bernoulli_distribution(0.25)(prng_))
          if (random::fast(uniform_distribution<bool>(0.25)))
          {
            new_peaks.erase(new_peaks.begin() + static_cast<std::int64_t>(change_index));
          }
          //else if (std::bernoulli_distribution(0.1)(prng_))
          else if (random::fast(uniform_distribution<bool>(0.1)))
          {
            new_peaks[change_index].position += random::fast(uniform_distribution(-span/100, span/100));
          }
          //else if (std::bernoulli_distribution(0.5)(prng_))
          else if (random::fast<bool>())
          {
            new_peaks[change_index].magnitude *= random::fast(uniform_distribution(0.5, 2.0));
          }
          else
          {
            new_peaks[change_index].width *= random::fast(uniform_distribution(0.5, 2.0));
          }
        }
        else
        {
          /*
          new_peaks.emplace_back(random::fast(uniform_distribution(min, max)), 
                                 random::fast(uniform_distribution(0.0, static_cast<double>(vec.size()) * pow<2>(spread) / bucket_count)),
                                 random::fast(uniform_distribution(2.5, span)));
                                 */

          const auto residuals = [&]
          {
            std::array<double, bucket_count> out;

            for (std::uint32_t i = 0; i < out.size(); ++i)
            {
              out[i] = distribution[i];

              for (const peak_t& peak : new_peaks)
              {
                const double offset = std::abs(peak.position - dist_vals[i]) / peak.width;

                out[i] -= peak.magnitude * std::exp(-pow<2>(offset));
              }
            }

            return out;
          }();

          const auto max_idx = static_cast<std::uint32_t>(std::max_element(residuals.begin(), residuals.end()) - residuals.begin());

          const double position = dist_vals[max_idx];

          const double magnitude = residuals[max_idx];

          double best_fit = std::numeric_limits<double>::infinity();

          double best_width = random::fast(uniform_distribution(2.5, span));

          for (auto _ = 500; _--;)
          {
            const double width = best_width * random::fast(uniform_distribution(9.0/10.0, 10.0/9.0));

            double fit{};

            for (std::uint32_t i = 0; i < bucket_count; ++i)
            {
              const double offset = std::abs(position - dist_vals[i]) / width;

              const double val = magnitude * std::exp(-pow<2>(offset));

              fit += pow<2>(val - residuals[i]);
            }

            if (fit < best_fit)
            {
              best_fit = fit;

              best_width = width;
            }
          }

          if (best_fit == std::numeric_limits<double>::infinity())
            fmt::print(fmt::emphasis::bold | fg(fmt::color::red), "ERROR: no change in best_fit when attempting to add a new curve\n");
          else
            new_peaks.emplace_back(position, magnitude, best_width);
        }

        return new_peaks;
      };

      auto evaluate_fit = [&](const auto new_peaks)
      {
        double fit{};

        for (std::uint32_t i = 0; i < distribution.size(); ++i)
        {
          double val{};

          for (const peak_t& peak : new_peaks)
          {
            const double offset = std::abs(peak.position - dist_vals[i]) / peak.width;

            val += peak.magnitude * std::exp(-pow<2>(offset));
          }

          fit += pow<2>(val - distribution[i]);
        }

        return fit;
      };

      double best_fit{std::numeric_limits<double>::infinity()};

      for (int i = 0; i < 50000; ++i)
      //for (int i = 0; i < 200000; ++i)
      //for (int i = 0; i < 5000; ++i)
      {
        const auto new_peaks = generate_new_peaks();

        const double new_fit = evaluate_fit(new_peaks);

        if (new_fit >= best_fit)
          continue;

        best_fit = new_fit;

        peaks = std::move(new_peaks);
      }

      fmt::print("{} peaks:\n", peaks.size());

      for (const peak_t& peak : peaks)
        fmt::print("{}\n", peak);


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

          mgraph.Add(graph.release());
        }

        // graph marking centers of underlying gaussians
        for (const peak_t& peak : peaks)
        {
          auto graph{std::make_unique<TGraph>()};

          graph->SetLineColor(kBlack);
          graph->SetLineWidth(4);

          graph->AddPoint(peak.position, 0);
          graph->AddPoint(peak.position, peak.magnitude);

          mgraph.Add(graph.release());
        }

        mgraph.Draw("a");

        canvas->SaveAs(fmt::format("cache/graph_{}.png", cols[n]).c_str());
      }

      fmt::print("finished with variable {} ({}/{})\n\n", cols[n], n + 1, cols.size());
    }
  };

  {
    std::vector<std::jthread> loop_threads{};

    for (std::uint32_t i = 1; i < std::thread::hardware_concurrency(); ++i)
      loop_threads.emplace_back(loop);

    fmt::print("Spawned {} threads.\n", loop_threads.size());
  }
}

int main()
{
  fit();
}

