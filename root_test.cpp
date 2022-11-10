#include "root.hpp"


enum class root_type { rt_none, rt_uint8, rt_uint16, rt_uint32, rt_uint64,
                                rt_int8, rt_int16, rt_int32, rt_int64,
                                rt_float, rt_double };


template<class T>
bool get_data(const movency::root::file& r, std::string name)
{
  auto data = r.uncompress<T>(name);

  fmt::print("found: {} entries for: {}\n", data.size(), name);

  for (const auto& d : data)
    fmt::print(FMT_COMPILE("{}\n"), d);

  return data.empty() ? false : true;
}


int main(int argc, char* argv[])
{
  if (argc < 3)
  {
    fmt::print("usage: {} <root_file_name> [--dump | --list | entry_name [type=double]]\n\n", argv[0]);
    fmt::print("  entry_name [type] to output all the data for that entry assuming it is encoded as type\n");
    fmt::print("    types are: uint8 uint16 uint32 uint64 int8 int16 int32 int64 float double\n\n");
    fmt::print("  --dump to output the TKey records in a root file\n");
    fmt::print("  --list to output the names in a root file with the total uncompressed bytes available\n");

    return EXIT_FAILURE;
  }

  std::string path = std::string(argv[1]);
  std::string name = std::string(argv[2]);

  root_type t = root_type::rt_none;

  if (argc == 4)
  {
    std::string t_str = std::string(argv[3]);

    if (t_str == "uint8")  t = root_type::rt_uint8;
    if (t_str == "uint16") t = root_type::rt_uint16;
    if (t_str == "uint32") t = root_type::rt_uint32;
    if (t_str == "uint64") t = root_type::rt_uint64;
    if (t_str == "int8")   t = root_type::rt_int8;
    if (t_str == "int16")  t = root_type::rt_int16;
    if (t_str == "int32")  t = root_type::rt_int32;
    if (t_str == "int64")  t = root_type::rt_int64;
    if (t_str == "float")  t = root_type::rt_float;
    if (t_str == "double") t = root_type::rt_double;

    if (t == root_type::rt_none)
    {
      fmt::print("Bad type provided: {}\n", t_str);

      return EXIT_FAILURE;
    }
  }
  else
    t = root_type::rt_double;

  movency::root::file r(path);

  if (!r.ok())
  {
    fmt::print("Unable to open root file: {}\n", path);

    return EXIT_FAILURE;
  }

  if (name == "--dump")
  {
    r.print_header();   // prints the file header
    r.print_contents(); // prints the top level tkey listing

    return EXIT_SUCCESS;
  }

  r.load_index(); // need to do this once (scans more/or/less the root file )-; could store this to another file and reuse...

  if (name == "--list")
  {
    const auto names = r.get_names();

    for (const auto& n : names)
      fmt::print(FMT_COMPILE("{}, {}\n"), n.first, n.second);

    return EXIT_SUCCESS;
  }

  bool ok = false;

  switch (t)
  {
    case root_type::rt_uint8:  ok = get_data<std::uint8_t> (r, name); break;
    case root_type::rt_uint16: ok = get_data<std::uint16_t>(r, name); break;
    case root_type::rt_uint32: ok = get_data<std::uint32_t>(r, name); break;
    case root_type::rt_uint64: ok = get_data<std::uint64_t>(r, name); break;
    case root_type::rt_int8:   ok = get_data<std::int8_t>  (r, name); break;
    case root_type::rt_int16:  ok = get_data<std::int16_t> (r, name); break;
    case root_type::rt_int32:  ok = get_data<std::int32_t> (r, name); break;
    case root_type::rt_int64:  ok = get_data<std::int64_t> (r, name); break;
    case root_type::rt_float:  ok = get_data<float>        (r, name); break;
    case root_type::rt_double: ok = get_data<double>       (r, name); break;
    default:
      ;
  }

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
