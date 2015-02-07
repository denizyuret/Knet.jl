function mysave(file, name, var)
h5create(file, name, size(var), 'Datatype', 'single', 'Deflate', 6, 'ChunkSize', size(var))
h5write (file, name, var)
end
