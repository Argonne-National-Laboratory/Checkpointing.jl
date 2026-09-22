"""
    HDF5Storage{MT}(checkpoints::Integer; filename = tempname())

A storage type for checkpointing that uses HDF5 files to store the checkpoints.

Requires the HDF5 package: do `using HDF5` before creating one. The type is
defined here so that `storage = HDF5Storage` can be passed to a scheme, but
everything that touches an HDF5 file lives in the `CheckpointingHDF5Ext`
package extension, which loads automatically once HDF5 is.
"""
mutable struct HDF5Storage{MT,F} <: AbstractStorage
    fid::F
    filename::String
    acp::Int64
end

# The two hooks the extension fills in. The fallbacks take `Any` so that the
# extension's methods, which constrain the argument types, are strictly more
# specific and add to these rather than overwrite them.
_hdf5_open(filename) = _hdf5_unavailable()
_hdf5_close(fid) = _hdf5_unavailable()

@noinline function _hdf5_unavailable()
    throw(
        ArgumentError(
            "[Checkpointing.jl]: HDF5Storage needs the HDF5 package. " *
            "Add `using HDF5` before creating one.",
        ),
    )
end

function HDF5Storage{Nothing}(acp::Integer; filename = tempname())
    # A placeholder never holds checkpoints, but opening the file here means a
    # missing HDF5 surfaces when the scheme is built, not deep inside `autodiff`.
    fid = _hdf5_open(filename)
    storage = HDF5Storage{Nothing,typeof(fid)}(fid, filename, acp)
    _hdf5_close(fid)
    return storage
end

function HDF5Storage{MT}(acp::Integer; filename = tempname()) where {MT}
    fid = _hdf5_open(filename)
    storage = HDF5Storage{MT,typeof(fid)}(fid, filename, acp)
    finalizer(s -> _hdf5_close(s.fid), storage)
    return storage
end

function Base.similar(storage::HDF5Storage, ::Type{T}) where {T}
    # A prototype can be instantiated more than once -- nested schemes share one
    # prototype object, and every reverse sweep instantiates again -- so each
    # instantiation needs its own file. Reopening `storage.filename` in "w" mode
    # would truncate the sibling scheme's checkpoints out from under it.
    return HDF5Storage{T}(storage.acp; filename = _sibling_filename(storage.filename))
end

function _sibling_filename(base::AbstractString)
    dir, file = splitdir(base)
    root, ext = splitext(file)
    return joinpath(dir, string(root, "_", string(rand(UInt32); base = 16), ext))
end
