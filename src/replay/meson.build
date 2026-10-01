# dxmt-replay.exe (MacNeutron): rebuilds a game's recorded D3D12 pipelines; see dxmt_replay.cpp.
executable('dxmt-replay', [ 'dxmt_replay.cpp', '../d3d12/d3d12_pipeline_record.cpp' ],
  dependencies        : [ util_dep ],
  include_directories : [ dxmt_include_path, include_directories('../d3d12') ],
  link_args           : [ '-municode' ],
  install             : true,
  install_dir         : windows_native_install_dir,
)
