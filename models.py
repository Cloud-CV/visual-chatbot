class VisDialDummyModel:

    def __init__(self, inputJson, loadPath, beamSize, beamLen, sampleWords,
                 temperature, gpuid, backend, proto_file, model_file,
                 maxThreads, encoder, decoder):
        print("init visdial model")
        self.decoder = decoder
        self.encoder = encoder
        self.maxThreads = maxThreads
        self.proto_file = proto_file
        self.model_file = model_file
        self.backend = backend
        self.gpuid = gpuid
        self.sampleWords = sampleWords
        self.beamLen = beamLen
        self.beamSize = beamSize
        self.loadPath = loadPath
        self.inputJson = inputJson
        self.temperature = temperature

    def predict(self, img, history, question):
        print("predict-visdial called!")
        print("img: ", img)
        print("hist: ", history)
        print("ques: ", question)
        dummy_ans_str = "dummy ans here to your dummy question there"
        dummy_hist_str = question + " " + dummy_ans_str

        result = {'answer': dummy_ans_str,
                  'question': question,
                  'history': ''.join(history) + dummy_hist_str, 'input_image': img}
        print(result)
        return result


class CaptioningTorchDummyModel:

    def __init__(self, model_path, backend, input_sz, layer, seed, gpuid):
        print("init caption model")
        self.seed = seed
        self.layer = layer
        self.input_sz = input_sz
        self.model_path = model_path
        self.backend = backend
        self.gpuid = gpuid
        self.loadModel(model_path)

    def loadModel(self, model_path):
        print("load-model called!")
        print("model_path: ", model_path)

    def predict(self, input_image_path, input_sz1, input_sz2):
        print("predict-caption called!")
        dummy_caption_str = "dummy caption here"
        result = {'input_image': input_image_path,
                  'pred_caption': dummy_caption_str}
        return result