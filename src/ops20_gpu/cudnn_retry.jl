using CUDA: CUDNN

# Repeat cudnn calls that fail due to memory issues
macro cudnn_retry(ex)
    quote
        res = CUDNN.@retry_reclaim x->(x ∈ (CUDNN.CUDNN_STATUS_ALLOC_FAILED, CUDNN.CUDNN_STATUS_EXECUTION_FAILED)) begin
            $ex
        end 
        if res != CUDNN.CUDNN_STATUS_SUCCESS
            CUDNN.throw_api_error(res)
        end
    end |> esc
end
