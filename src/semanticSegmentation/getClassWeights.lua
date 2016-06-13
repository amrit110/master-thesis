--This script is useful for obtaining mini-batch histograms so that class
--weights can be set properly when the data is not uniformly represented across
--classes

--dofile 'data_semantic_second_stage.lua'
dofile 'data.lua'

N = 50
nClasses = 20
results = torch.zeros(N,nClasses)
for i = 1,N do
    local imgs, labs = loadMiniBatch(100,'train')
    local hist = torch.histc(labs,nClasses):add(1)
    hist = hist:div(torch.max(hist))
    local tmp = torch.ones(nClasses)
    results[i] = torch.cdiv(tmp,hist)
    finalResults = torch.mean(results,1)
    xlua.progress(i,N)
end

t1 = {}
for i = 1,finalResults:size(1) do
    table.insert(t1,finalResults[i])
end



