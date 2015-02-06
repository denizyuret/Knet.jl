function msg(varargin)
if gpuDeviceCount > 0
  g = gpuDevice;
  fmt = ['%s (gpu:%.2e) ' varargin{1} '\n'];
  feval(@fprintf, fmt, datestr(now), g.FreeMemory, varargin{2:end});
else
  fmt = ['%s ' varargin{1} '\n'];
  feval(@fprintf, fmt, datestr(now), varargin{2:end});
end
end
