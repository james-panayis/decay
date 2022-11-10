#pragma once

#include "timeblit/read_from.hpp"

#include "fmt/format.h"
#include "fmt/compile.h"

#include <zlib/zlib.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <vector>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <array>

namespace std {

template<movency::integral T>
constexpr T byteswap(const T n) noexcept
{
  const auto in_arr = std::bit_cast<std::array<std::uint8_t, sizeof(T)>>(n);

  std::remove_const_t<decltype(in_arr)> out_arr;

  for (std::size_t i = 0; i < in_arr.size(); ++i)
    out_arr[i] = in_arr[sizeof(T) - 1 - i];

  return std::bit_cast<T>(out_arr);
}

template<class Enum>
  requires is_enum_v<Enum>
constexpr auto to_underlying(const Enum e) noexcept
{
  return static_cast<std::underlying_type_t<Enum>>(e);
}

}


namespace movency {
namespace root {

  // utility template to provide a uint of a fixed size

  template<std::size_t W>
  using uint_of_width = std::conditional_t<(W == 1), std::uint8_t,
                        std::conditional_t<(W == 2), std::uint16_t,
                        std::conditional_t<(W == 4), std::uint32_t, std::uint64_t>>>;

  // utility functions to read data from memory

  template<class T>
  inline bool read_from_and_subspan(T& t, std::span<const std::byte>& buf) noexcept
  {
    if (buf.size() < sizeof(T))
      return false;

    t = read_from(buf.data());

    buf = buf.subspan(sizeof(T));

    return true;
  }

  // reading string_view's

  template<>
  inline bool read_from_and_subspan(std::string_view& t, std::span<const std::byte>& buf) noexcept
  {
    std::uint8_t len;

    if (!read_from_and_subspan(len, buf))
      return false;

    if (len > buf.size())
      return false;

    t = std::string_view(reinterpret_cast<const char*>(buf.data()), len);

    buf = buf.subspan(len);

    return true;
  }


  // root stores data (usually!) in Big Endian form, so we need to byteswap to convert values to native Little Endian form

  template<class T>
  inline bool read_from_be_and_subspan(T& t, std::span<const std::byte>& buf) noexcept
  {
    if (read_from_and_subspan(t, buf))
    {
      t = std::byteswap(t);
      return true;
    }
    else
      return false;
  }


  // A root file has the following header

  struct header
  {
    std::array<char, 4>          ident;        // Root file identifier
    std::uint32_t                fVersion;     // File format version
    std::uint32_t                fBEGIN;       // Pointer to first data record
    std::uint64_t                fEND;         // Pointer to first free word at the EOF
    std::uint64_t                fSeekFree;    // Pointer to FREE data record
    std::uint32_t                fNbytesFree;  // Number of bytes in FREE data record
    std::uint32_t                nfree;        // Number of free data records
    std::uint32_t                fNbytesName;  // Number of bytes in TNamed at creation time
    std::uint8_t                 fUnits;       // Number of bytes for file pointers
    std::uint32_t                fCompress;    // Compression level and algorithm
    std::uint32_t                fSeekInfo;    // Pointer to TStreamerInfo record
    std::uint32_t                fNbytesInfo;  // Number of bytes in TStreamerInfo record
    std::array<std::uint8_t, 16> fUUID;        // Universal Unique ID

    bool ok{false};


    header() noexcept
    {}


    header(const std::span<const std::byte> source) noexcept
    {
      load(source);
    }


    bool load(std::span<const std::byte> source) noexcept
    {
      ok = true;
      
      ok &= read_from_and_subspan   (ident,       source);
      ok &= read_from_be_and_subspan(fVersion,    source);
      ok &= read_from_be_and_subspan(fBEGIN,      source);
      ok &= read_from_be_and_subspan(fEND,        source);
      ok &= read_from_be_and_subspan(fSeekFree,   source);
      ok &= read_from_be_and_subspan(fNbytesFree, source);
      ok &= read_from_be_and_subspan(nfree,       source);
      ok &= read_from_be_and_subspan(fNbytesName, source);
      ok &= read_from_and_subspan   (fUnits,      source);
      ok &= read_from_be_and_subspan(fCompress,   source);
      ok &= read_from_be_and_subspan(fSeekInfo,   source);
      ok &= read_from_be_and_subspan(fNbytesInfo, source);
      ok &= read_from_and_subspan   (fUUID,       source);

      return ok;
    }
  };


  // A root file is a list of tkey records

  struct tkey
  {
    std::int32_t               Nbytes;    // Length of compressed object (in bytes)
    std::uint16_t              Version;   // TKey version identifier
    std::uint32_t              ObjLen;    // Length of uncompressed object
    std::uint32_t              Datime;    // Date and time when object was written to file
    std::uint16_t              KeyLen;    // Length of the key structure (in bytes)
    std::uint16_t              Cycle;     // Cycle of key
    std::uint64_t              SeekKey;   // Pointer to record itself (consistency check)
    std::uint64_t              SeekPdir;  // Pointer to directory header

    std::string_view           ClassName; // The type of record this is
    std::string_view           Name;      // The Name of this entry
    std::string_view           Title;     // An optional Title
    std::span<const std::byte> DATA;      // The data store for this record

    std::uint64_t base;
    bool          ok{false};

    tkey() noexcept
    {}


    tkey(std::span<const std::byte> source, const std::uint64_t pos) noexcept
    {
      load(source, pos);
    }


    bool load(std::span<const std::byte> source, const std::uint64_t pos) noexcept
    {
      base = pos;
      ok = true;

      source = source.subspan(pos);

      const auto start = source.data();

      ok &= read_from_be_and_subspan(Nbytes,   source);

      if (Nbytes < 0) // record is marked as deleted
        return ok;

      ok &= read_from_be_and_subspan(Version,  source);
      ok &= read_from_be_and_subspan(ObjLen,   source);
      ok &= read_from_be_and_subspan(Datime,   source);
      ok &= read_from_be_and_subspan(KeyLen,   source);
      ok &= read_from_be_and_subspan(Cycle,    source);

      if (Version > 1'000)
      {
        ok &= read_from_be_and_subspan(SeekKey,  source);
        ok &= read_from_be_and_subspan(SeekPdir, source);
      }
      else
      {
        std::uint32_t SeekKey_u32;
        std::uint32_t SeekPdir_u32;

        ok &= read_from_be_and_subspan(SeekKey_u32,  source);
        ok &= read_from_be_and_subspan(SeekPdir_u32, source);

        SeekKey  = SeekKey_u32;
        SeekPdir = SeekPdir_u32;
      }

      ok &= read_from_and_subspan(ClassName,   source);
      ok &= read_from_and_subspan(Name,        source);
      ok &= read_from_and_subspan(Title,       source);

      // Sometimes a record has more entry following Title which are part of the Key and the DATA part is after this

      const auto bytes_read = static_cast<std::uint16_t>(source.data() - start);

      DATA = source.subspan(KeyLen - bytes_read, static_cast<std::uint64_t>(Nbytes - KeyLen));

      return ok;
    }


    void print() const
    {
      if (Nbytes < 0)
        fmt::print("ok: {} base: {} Nbytes: {}\n", ok, base, Nbytes);
      else
        fmt::print("ok: {} base: {} Nbytes: {} Version: {} ObjLen: {} Datime: {} KeyLen: {} Cycle: {} SeekKey: {} SeekPdir: {} ClassName: {} Name: {} Title: {} DATA.size: {}\n",
          ok, base, Nbytes, Version, ObjLen, Datime, KeyLen, Cycle, SeekKey, SeekPdir, ClassName, Name, Title, DATA.size());
    }
  };


  // a root compression header uses 9 bytes: 3 for compression algorithm, 3 for compressed size and 3 for uncompressed size
  // This is stored before the compressed data in the TBasket DATA

  struct compress_header
  {
    static constexpr int SIZE = 9;

    enum class engine { none, zlib, lzma, lz4, zstd };

    engine        e;
    std::uint32_t compressed_size;
    std::uint32_t uncompressed_size;


    compress_header(std::span<const std::byte>& source) noexcept
    {
      load(source);
    }


    engine get_engine(const unsigned char* src) noexcept
    {
      if (src[0] == 'Z' && src[1] == 'L' && static_cast<int>(src[2]) == Z_DEFLATED)
        return engine::zlib;
      
      if (src[0] == 'X' && src[1] == 'T' && src[2] == 0)
        return engine::lzma;
      
      if (src[0] == 'L' && src[1] == '4')
        return engine::lz4;
      
      if (src[0] == 'Z' && src[1] == 'S' && src[2] == 1)
        return engine::zstd;
      
      return engine::none;
    }


    bool load(std::span<const std::byte>& source) noexcept
    {
      if (source.size() < SIZE)
      {
        e = engine::none;
        return false;
      }

      auto p = reinterpret_cast<const unsigned char*>(source.data());

      e = get_engine(p);

      // NB: These are is Little endian form! Compared to the rest of the file which is in big endian form.

      compressed_size   = p[3]  | (p[4] << 8) | (p[5] << 16);
      uncompressed_size = p[6]  | (p[7] << 8) | (p[8] << 16);

      source = source.subspan(SIZE);

      return e != engine::none;
    }
  };


  class file
  {
  public:

    file(std::string path) noexcept
    {
      open(path);
    }


    bool ok() const noexcept
    {
      return !file_.empty();
    }


    ~file() noexcept
    {
      close();
    }


    auto size() const noexcept
    {
      return file_.size();
    }


    bool open(std::string_view path) noexcept
    {
      path_ = path;

      const auto fd = ::open(path_.c_str(), O_RDONLY);

      if (fd == -1)
      {
        fmt::print("cannot open root file: {}\n", path_);
        return false;
      }

      struct stat sb;

      if (fstat(fd, &sb) == -1)
      {
        fmt::print("cannot stat root file: {}\n", path_);
        ::close(fd);
        return false;
      }
      
      auto data_start = reinterpret_cast<const std::byte*>(mmap(NULL, sb.st_size, PROT_READ, MAP_SHARED, fd, 0));

      ::close(fd);

      if (data_start == reinterpret_cast<const std::byte*>(-1))
      {
        fmt::print("cannot map root file: {}\n", path_);
        return false;
      }

      file_ = std::span<const std::byte>{ data_start, static_cast<std::size_t>(sb.st_size) };

      if (!h_.load(file_))
      {
        fmt::print("Unable to load root header: {}\n", path_);
        close();
        return false;
      }

      return true;
    }


    tkey load_tkey(std::uint64_t pos) const noexcept
    {
      return tkey(file_, pos);
    }


    bool load_index() noexcept
    {
      std::uint64_t pos = h_.fBEGIN;

      auto t = load_tkey(pos);
      
      while (t.ok)
      {
        if (t.ClassName == "TBasket")
        {
          auto& m = baskets_[std::string{t.Name}];

          m.cycles[t.Cycle] = t.base;
          m.total_bytes    += t.ObjLen;
        }

        pos += static_cast<std::uint64_t>(abs(t.Nbytes)); // -ve indicates a deleted entry
        
        if (pos >= size())
          return true;
      
        t = load_tkey(pos);
      }

      return false;
    }


    void print_index() const noexcept
    {
      for (const auto& [ lhs, rhs] : baskets_)
      {
        fmt::print("name: {} total_bytes: {} offsets:", lhs, rhs.total_bytes);

        for (const auto& [ cycle, offset ] : rhs.cycles)
          fmt::print(" {} -> {},", cycle, offset);

        fmt::print("\n");
      }
    }


    void print_contents() const noexcept
    {
      std::uint64_t pos = h_.fBEGIN;

      auto t = load_tkey(pos);

      while (t.ok)
      {
        t.print();

        pos += static_cast<std::uint64_t>(abs(t.Nbytes)); // -ve indicates a deleted entry
        
        if (pos >= size())
          break;

        t = load_tkey(pos);
      }
    }


    std::vector<std::pair<std::string_view, std::uint64_t>> get_names() const noexcept
    {
      std::vector<std::pair<std::string_view, std::uint64_t>> names;

      names.reserve(baskets_.size());

      for (const auto& [ n, b] : baskets_)
        names.emplace_back(n, b.total_bytes);

      return names;
    }


    // uncompress a record

    template<class T>
    bool uncompress(std::span<T> dest, const tkey& t) const noexcept
    {
      std::uint64_t dest_len = dest.size_bytes();

      if (dest_len != t.ObjLen)
      {
        fmt::print("Bad output size, provided: {} wanted: {}\n", dest_len, t.ObjLen);
        return false;
      }

      auto src = t.DATA;

      compress_header h(src);

      if (h.e == compress_header::engine::none)
      {
        fmt::print("Bad compress engine\n");
        return false;
      }

      if (h.compressed_size != src.size())
      {
        fmt::print("bad compressed size");
        return false;
      }

      if (h.uncompressed_size != t.ObjLen)
      {
        fmt::print("bad uncompressed size");
        return false;
      }

      int err;

      if (h.e == compress_header::engine::zlib)
      {
        err = ::uncompress(reinterpret_cast<unsigned char*>(dest.data()), &dest_len,
                           reinterpret_cast<const unsigned char*>(src.data()), src.size());
      }
      else
      {
        fmt::print("compression type not supported: '{}'\n", std::to_underlying(h.e));
        return false;
      }

      if ((err == Z_OK) && (dest_len == dest.size_bytes())) // uncompress is good
      {
        // byteswap the values to convert from Big Endian to native Little Endian

        for (auto& d : dest)
          d = std::bit_cast<T>(std::byteswap(std::bit_cast<uint_of_width<sizeof(T)>>(d))); // can only byteswap u/ints, so bitcast

        return true;
      }
      else
        return false;
    }


    // uncompress all records for a matching Name

    template<class T>
    std::vector<T> uncompress(std::string_view id) const noexcept
    {
      const auto it = baskets_.find(id);

      if (it == baskets_.end())
      {
        fmt::print("Unable to find baskets for: {}\n", id);
        return {};
      }

      const std::uint64_t total_entries = it->second.total_bytes / sizeof(T);

      std::vector<T> r(total_entries);

      std::uint64_t pos = 0;

      std::span<T> build = r;

      for (const auto& [c, offset] : it->second.cycles)
      {
        auto t = load_tkey(offset);

        if (!t.ok)
        {
          fmt::print("Found a bad record at index: {}\n", c);
          return {};
        }

        auto entries = t.ObjLen / sizeof(T);

        if (!uncompress(build.subspan(pos, entries), t))
        {
          fmt::print("Unable to uncompress index: {}\n", c);
          return {};
        }

        pos += entries;
      }

      return r;
    }


    // start of parsing TFile record entries

    void explore_tfile(const tkey& t) const noexcept
    {
      if (t.Nbytes < 0)
        return;

      auto source = t.DATA;

      std::uint16_t version;
      std::uint32_t DatimeC;
      std::uint32_t DatimeM;
      std::uint32_t NbytesKeys;
      std::uint32_t NbytesName;

      std::uint64_t SeekDir;
      std::uint64_t SeekParent;
      std::uint64_t SeekKeys;
      std::uint16_t UUIDver;
      std::array<uint8_t, 16> UUID;

      bool ok = true;

      ok &= read_from_be_and_subspan(version,    source);
      ok &= read_from_be_and_subspan(DatimeC,    source);
      ok &= read_from_be_and_subspan(DatimeM,    source);
      ok &= read_from_be_and_subspan(NbytesKeys, source);
      ok &= read_from_be_and_subspan(NbytesName, source);

      if (t.Version > 1000)
      {
        ok &= read_from_be_and_subspan(SeekDir,    source);
        ok &= read_from_be_and_subspan(SeekParent, source);
        ok &= read_from_be_and_subspan(SeekKeys,   source);
      }
      else
      {
        std::uint32_t SeekDir_u32;
        std::uint32_t SeekParent_u32;
        std::uint32_t SeekKeys_u32;

        ok &= read_from_be_and_subspan(SeekDir_u32,    source);
        ok &= read_from_be_and_subspan(SeekParent_u32, source);
        ok &= read_from_be_and_subspan(SeekKeys_u32,   source);

        SeekDir = SeekDir_u32;
        SeekParent = SeekParent_u32;
        SeekKeys = SeekKeys_u32;
      }

      ok &= read_from_be_and_subspan(UUIDver, source);
      ok &= read_from_and_subspan   (UUID,    source);

      if (ok)
        fmt::print("all done, version: {} DatimeC: {}, DatimeM: {} NbytesKeys: {} NbytesName: {} SeekDir: {} SeekParent: {} SeekKeys: {} UUIDver: {}\n",
          version, DatimeC, DatimeM, NbytesKeys, NbytesName, SeekDir, SeekParent, SeekKeys, UUIDver);
      else
        fmt::print("got bad TDirectory\n");
    }


    // start of parsing KeysList record entries

    void explore_keyslist(const tkey& t) const noexcept
    {
      if (t.Nbytes < 0)
        return;

      auto source = t.DATA;

      std::uint32_t NKeys;

      bool ok = true;

      ok &= read_from_be_and_subspan(NKeys, source);

      if (ok)
      {
        for (std::uint32_t i = 0; i < NKeys; ++i)
        {
          auto t2 = load_tkey(static_cast<std::uint64_t>(source.data() - file_.data()));

          t2.print();
        }

        fmt::print("all done, NKeys: {}\n", NKeys);
        return;
      }

      fmt::print("got bad keyslist\n");
    }


    void print_header() const noexcept
    {
      fmt::print("ident: {} fVersion: {} fBEGIN: {} fEND: {} fSeekFree: {} fNbytesFree: {} nfree: {} fNbytesName: {} fUnits: {} fCompress: {} fSeekInfo: {} fNbytesInfo: {}\n",
        std::string_view(h_.ident.data(), h_.ident.size()), h_.fVersion, h_.fBEGIN, h_.fEND, h_.fSeekFree, h_.fNbytesFree, h_.nfree, h_.fNbytesName, h_.fUnits, h_.fCompress, h_.fSeekInfo, h_.fNbytesInfo);
    }


    void close() noexcept
    {
      if (!file_.empty())
      {
        munmap((void*)file_.data(), file_.size());
        file_ = {};
      }
    }


    header h_;


  private:


    std::string path_;

    std::span<const std::byte> file_;

    // For a particular measure, eg: h1_PX, there are a number of TBasket's with ascending Cycle identifiers.
    // These are the compressed data, so store them here with the offset to access.

    struct basket_info
    {
      std::uint64_t                total_bytes{0};  // total uncompressed bytes available for this (sum of all cycle TBasket's)
      std::map<int, std::uint64_t> cycles;          // map from cycle_id -> offset_in_file
    };

    std::map<std::string, basket_info, std::less<>> baskets_; // name -> baskets_for_this_item

  }; // class file

} // namespace root
} // namespace movency
