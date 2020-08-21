module GLUE
using Pkg.Artifacts

function download()
    artifact_toml = joinpath(@__DIR__, "Artifacts.toml")
    glue_hash = artifact_hash("glue", artifact_toml)
    if glue_hash === nothing || !artifact_exists(glue_hash)
        glue_hash = create_artifact() do artifact_dir
            download_glue_data = tempdir()*"/download_glue_data.py"
            Base.download("https://raw.githubusercontent.com/nyu-mll/jiant/master/scripts/download_glue_data.py", download_glue_data)
            run(`python $download_glue_data --data_dir $artifact_dir`)
            rm(download_glue_data)
        end
        bind_artifact!(artifact_toml, "glue", glue_hash)
    end
    return artifact_path(glue_hash)
end

end
