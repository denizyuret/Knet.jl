function find_compiler()
    # find a suitable host compiler
    if is_windows()
        # Windows: just use cl.exe
        vc_versions = ["VS140COMNTOOLS", "VS120COMNTOOLS", "VS110COMNTOOLS", "VS100COMNTOOLS"]
        !any(x -> haskey(ENV, x), vc_versions) && error("Compatible Visual Studio installation cannot be found; Visual Studio 2015, 2013, 2012, or 2010 is required.")
        vs_cmd_tools_dir = ENV[vc_versions[first(find(x -> haskey(ENV, x), vc_versions))]]
        cl_path = joinpath(dirname(dirname(dirname(vs_cmd_tools_dir))), "VC", "bin", Sys.WORD_SIZE == 64 ? "amd64" : "", "cl.exe")

        host_compiler = cl_path
        host_version = nothing
    else
        # Unix-like platforms: find compatible GCC binary

        # enumerate possible names for the gcc binary
        # NOTE: this is coarse, and might list invalid, non-existing versions
        gcc_names = [ "gcc" ]
        for major in 3:7
            push!(gcc_names, "gcc-$major")
            for minor in 0:9
                push!(gcc_names, "gcc-$major.$minor")
                push!(gcc_names, "gcc$major$minor")
            end
        end

        # find the binary
        gcc_possibilities = []
        for gcc_name in gcc_names
            # check if the binary exists
            gcc_path = try
                find_binary(gcc_name)
            catch ex
                isa(ex, ErrorException) || rethrow(ex)
                continue
            end

            # parse the GCC version string
            verstring = chomp(readlines(`$gcc_path --version`)[1])
            m = match(Regex("^$gcc_name \\(.*\\) ([0-9.]+)"), verstring)
            if m === nothing
                warn("Could not parse GCC version info (\"$verstring\"), assuming v0.")
                gcc_ver = v"0"
            else
                gcc_ver = VersionNumber(m.captures[1])
            end
            push!(gcc_possibilities, (gcc_path, gcc_ver))
        end

        # select the most recent compiler
        if length(gcc_possibilities) == 0
            if is_apple()
                # GCC is no longer supported on MacOS so let's just use clang
                # TODO: proper version matching, etc
                clang_path = find_binary("clang")

                host_compiler = clang_path
                host_version = nothing
            else
                error("Could not find a suitable host compiler")
            end
        else
            sort!(gcc_possibilities; rev=true, lt=(a, b) -> a[2]<b[2])
            host_compiler, host_version = gcc_possibilities[1]
        end
    end
    return host_compiler
end

function find_binary(name, prefixes::Vector{String}=String[])
    # figure out names
    if is_windows()
        name = "$name.exe"
    end

    # figure out locations
    locations = []
    for prefix in prefixes
        push!(locations, prefix)
        push!(locations, joinpath(prefix, "bin"))
    end
    let path = ENV["PATH"]
        dirs = split(path, is_windows() ? ';' : ':')
        filter!(path->!isempty(path), dirs)
        append!(locations, dirs)
    end

    paths = [joinpath(location, name) for location in locations]
    try
        paths = filter(ispath, paths)
    end
    paths = unique(paths)
    if isempty(paths)
        error("Could not find $name binary")
    end

    path = first(paths)
    return path
end
