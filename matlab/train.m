function net = train(net, x, y, varargin)
    o = options(net, x, y, varargin{:});
    r = initreport(net, o);
    M = size(x, 2);
    L = numel(net);
    E = o.epochs;
    B = o.batch;

    for e = 1:E
        for i = 1:B:M
            j = min(i+B-1, M);

            a = x(:,i:j);
            for l=1:L
                if net{l}.dropout
                    a = net{l}.drop(a);
                end
                a = net{l}.forw(a);
            end

            d = y(:,i:j);
            for l=L:-1:2
                d = net{l}.back(d);
            end
            % last dx is slow and unnecessary
            net{1}.back(d);

            for l=1:L
                net{l}.update();
            end

            r = report(net, o, r);
        end
    end
    finalreport(net, o, r);
end


function r = report(net, o, r)
    r.instances = r.instances + size(net{1}.x, 2);
    if o.test >= 1 && r.instances >= r.nexttest
        if r.instances == 0
            fprintf('%-13s', 'inst');
            for i=1:2:numel(o.testdata)
                fprintf('%-13s%-13s', 'loss', 'acc');
            end
            fprintf('%-13s%-13s\n', 'speed', 'time');
        end
        r.nexttest = r.nexttest + o.test;
        fprintf('%-13d', r.instances);
        for i=1:2:numel(o.testdata)
            [acc, loss] = evalnet(net, o.testdata{i}, o.testdata{i+1});
            fprintf('%-13g%-13g', loss, acc);
        end
        fprintf('%-13g%-13g\n', r.instances/toc(r.time), toc(r.time));
    end
    if o.stats >= 1 && r.instances >= r.nextstat
        r.nextstat = r.nextstat + o.stats;
        fprintf('\n%-13s%-13s%-13s%-13s%-13s%-13s\n', 'array', 'min', ...
                'rms', 'max', 'nz', 'nzrms');
        for l=1:numel(net)
            summary(net, l, 'x');
            summary(net, l, 'y');
            summary(net, l, 'w');
            summary(net, l, 'dw');
            summary(net, l, 'dw1');
            summary(net, l, 'dw2');
            fprintf('\n');
        end
    end
    if o.save >= 1 && r.instances >= r.nextsave
        fname = sprintf('%s%d.mat', o.savename, r.instances);
        fprintf('Saving %s...', fname);
        save(fname, 'net', '-v7.3');
        fprintf('done\n');
        if r.nextsave == 0
            r.nextsave = o.save;
        else
            r.nextsave = r.nextsave * 2;
        end
    end
end

function r = initreport(net, o)
    r.time = tic;
    r.instances = 0;
    r.nexttest = 0;
    r.nextstat = 0;
    r.nextsave = 0;
    r = report(net, o, r);
end

function finalreport(net, o, r)
    net{1}.x = [];
    r.nexttest = 0;
    r.nextstat = 0;
    r.nextsave = 0;
    report(net, o, r);
end

function summary(net, l, f)
    a = getfield(net{l}, f);
    if isempty(a) return; end
    nm = sprintf('n%d.%s', l, f);
    nz = (a(:)~=0);
    fprintf('%-13s%-13g%-13g%-13g%-13g%-13g\n', ...
            nm, min(a(:)), sqrt(mean(a(:).^2)), max(a(:)), ...
            mean(nz), sqrt(mean(a(nz).^2)));
end

function o = options(net, x, y, varargin)
    p = inputParser;
    p.addRequired('net', @iscell);
    p.addRequired('x', @isnumeric);
    p.addRequired('y', @isnumeric);
    p.addParamValue('epochs', 1, @isnumeric);
    p.addParamValue('batch', 100, @isnumeric);
    p.addParamValue('test', 0, @isnumeric);
    p.addParamValue('testdata', {}, @iscell);
    p.addParamValue('save', 0, @isnumeric);
    p.addParamValue('savename', 'net', @ischar);
    p.addParamValue('stats', 0, @isnumeric);
    p.parse(net, x, y, varargin{:});
    o = p.Results;
end
