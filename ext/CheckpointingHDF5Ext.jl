module CheckpointingHDF5Ext

# Everything that touches an HDF5 file. The `HDF5Storage` type itself, its
# constructors and `similar` live in Checkpointing so that `storage = HDF5Storage`
# works without this extension loaded; the two `_hdf5_*` hooks below are what
# the constructors call into.

using HDF5
using Checkpointing: Checkpointing, HDF5Storage, checkpoint_copy!
import Checkpointing: _hdf5_open, _hdf5_close, save!, load!

_hdf5_open(filename::AbstractString) = h5open(filename, "w")
_hdf5_close(fid::HDF5.File) = close(fid)

function save!(storage::HDF5Storage{MT}, v::MT, i::Integer) where {MT}
    @assert i >= 1 && i <= storage.acp
    if haskey(storage.fid, "$i")
        delete_object(storage.fid, "$i")
    end
    blob = Checkpointing.serialize(v)
    storage.fid["$i"] = blob
    return storage
end

function load!(body::MT, storage::HDF5Storage{MT}, i::Integer) where {MT}
    @assert i >= 1 && i <= storage.acp
    blob = read(storage.fid["$i"])
    return checkpoint_copy!(body, Checkpointing.deserialize(blob))
end

end
