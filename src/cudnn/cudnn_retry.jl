import CUDA

# Repeat cudnn calls that fail due to memory issues
macro cudnn_retry(ex)
    quote
        res = CUDA.CUDNN.@retry_reclaim x->(x ∈ (CUDA.CUDNN.CUDNN_STATUS_ALLOC_FAILED, CUDA.CUDNN.CUDNN_STATUS_EXECUTION_FAILED)) begin
            $ex
        end 
        if res != CUDA.CUDNN.CUDNN_STATUS_SUCCESS
            CUDA.CUDNN.throw_api_error(res)
        end
    end |> esc
end
