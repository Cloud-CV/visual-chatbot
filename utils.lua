-- script containing supporting code/methods
local utils = {};
cjson = require 'cjson'

-- right align the question tokens in 3d volume
function utils.rightAlign(sequences, lengths)
    -- clone the sequences
    local rAligned = sequences:clone():fill(0);
    local numDims = sequences:dim();

    if numDims == 3 then
        local M = sequences:size(3); -- maximum length of question
        local numImgs = sequences:size(1); -- number of images
        local maxCount = sequences:size(2); -- number of questions / image

        for imId = 1, numImgs do
            for quesId = 1, maxCount do
                -- do only for non zero sequence counts
                if lengths[imId][quesId] == 0 then
                    break;
                end

                -- copy based on the sequence length
                rAligned[imId][quesId][{{M - lengths[imId][quesId] + 1, M}}] =
                            sequences[imId][quesId][{{1, lengths[imId][quesId]}}];
            end
        end
    else if numDims == 2 then
        -- handle 2 dimensional matrices as well
        local M = sequences:size(2); -- maximum length of question
        local numImgs = sequences:size(1); -- number of images

        for imId = 1, numImgs do
            -- do only for non zero sequence counts
            if lengths[imId] > 0 then
                -- copy based on the sequence length
                rAligned[imId][{{M - lengths[imId] + 1, M}}] =
                        sequences[imId][{{1, lengths[imId]}}];
            end
        end
        end
    end

    return rAligned;
end

-- translate a table of words to index tensor
function utils.wordsToId(words, word2ind, max_len)
    local len = max_len or 15
    local vector = torch.LongTensor(len):zero()
    for i = 1, #words do
        if word2ind[words[i]] ~= nil then
            vector[len - #words + i] = word2ind[words[i]]
        else
            vector[len - #words + i] = word2ind['UNK']
        end
    end
    return vector
end

-- translate a given tensor/table to sentence
function utils.idToWords(vector, ind2word)
    local sentence = '';

    local nextWord;
    for wordId = 1, vector:size(1) do
        if vector[wordId] > 0 then
            nextWord = ind2word[vector[wordId]]; 
            sentence = sentence..' '..nextWord;
        end

        -- stop if end of token is attained
        if nextWord == '<END>' then break; end
    end
    
    return sentence;
end

-- read a json file and lua table
function utils.readJSON(fileName)
    local file = io.open(fileName, 'r');
    local text = file:read();
    file:close();
    
    -- convert and save information
    return cjson.decode(text);
end

-- save a lua table to the json
function utils.writeJSON(fileName, luaTable)
    -- serialize lua table
    local text = cjson.encode(luaTable)

    local file = io.open(fileName, 'w');
    file:write(text);
    file:close();
end

-- compute the likelihood given the gt words and predicted probabilities
function utils.computeLhood(words, predProbs)
    -- compute the probabilities for each answer, based on its tokens
    -- convert to 2d matrix
    local predVec = predProbs:view(-1, predProbs:size(3));
    local indices = words:contiguous():view(-1, 1);
    local mask = indices:eq(0);
    -- assign proxy values to avoid 0 index errors
    indices[mask] = 1;
    local logProbs = predVec:gather(2, indices);
    -- neutralize other values
    logProbs[mask] = 0;
    logProbs = logProbs:viewAs(words);
    -- sum up for each sentence
    logProbs = logProbs:sum(1):squeeze();

    return logProbs;
end

-- process the scores and obtain the ranks
-- input: scores for all options, ground truth positions
function utils.computeRanks(scores, gtPos)
    local gtScore = scores:gather(2, gtPos);
    local ranks = scores:gt(gtScore:expandAs(scores));
    ranks = ranks:sum(2) + 1;

    -- convert into double
    return ranks:double();
end

-- process the ranks and print metrics
function utils.processRanks(ranks)
    -- print the results
    local numQues = ranks:size(1) * ranks:size(2);

    local numOptions = 100;

    -- convert ranks to double, vector and remove zeros
    ranks = ranks:double():view(-1);
    -- non of the values should be 0, there is gt in options
    if torch.sum(ranks:le(0)) > 0 then
        numZero = torch.sum(ranks:le(0));
        print(string.format('Warning: some of ranks are zero : %d', numZero))
        ranks = ranks[ranks:gt(0)];
    end

    if torch.sum(ranks:ge(numOptions + 1)) > 0 then 
        numGreater = torch.sum(ranks:ge(numOptions + 1));
        print(string.format('Warning: some of ranks >100 : %d', numGreater))
        ranks = ranks[ranks:le(numOptions + 1)];
    end

    ------------------------------------------------
    print(string.format('\tNo. questions: %d', numQues))
    print(string.format('\tr@1: %f', torch.sum(torch.le(ranks, 1))/numQues))
    print(string.format('\tr@5: %f', torch.sum(torch.le(ranks, 5))/numQues))
    print(string.format('\tr@10: %f', torch.sum(torch.le(ranks, 10))/numQues))
    print(string.format('\tmedianR: %f', torch.median(ranks:view(-1))[1]))
    print(string.format('\tmeanR: %f', torch.mean(ranks)))
    print(string.format('\tmeanRR: %f', torch.mean(ranks:cinv())))
end

function utils.preprocess(path, width, height)
    local width = width or 224
    local height = height or 224

    -- load image
    local orig_image = image.load(path)

    -- handle greyscale and rgba images
    if orig_image:size(1) == 1 then
        orig_image = orig_image:repeatTensor(3, 1, 1)
    elseif orig_image:size(1) == 4 then
        orig_image = orig_image[{{1,3},{},{}}]
    end

    -- get the dimensions of the original image
    local im_height = orig_image:size(2)
    local im_width = orig_image:size(3)

    -- scale and subtract mean
    local img = image.scale(orig_image, width, height):double()
    local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
    img = img:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img:add(-1, mean_pixel)
    return img, im_height, im_width
end

return utils;
