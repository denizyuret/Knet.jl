module SuperGLUE
using Pkg.Artifacts

function download()
    artifact_toml = joinpath(@__DIR__, "Artifacts.toml")
    superglue_hash = artifact_hash("superglue", artifact_toml)
    if superglue_hash === nothing || !artifact_exists(superglue_hash)
        superglue_hash = create_artifact() do artifact_dir
            path = tempdir()*"/combined.zip"
            Base.download("https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip", path)
            run(`unzip -x $path -d $artifact_dir`)
            rm(path)
        end
        bind_artifact!(artifact_toml, "superglue", superglue_hash)
    end
    return artifact_path(superglue_hash)
end

end
