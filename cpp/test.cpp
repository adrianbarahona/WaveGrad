#include <torch/script.h> // One-stop header.
#include <iostream>
#include <memory>
#include "librosa.h"
#include "wavreader.h"

using namespace std; 


std::vector<float> read_wav(const std::string& path)
{
    void* h_x = wav_read_open(path.c_str());

    int format, channels, sr, bits_per_sample;
    unsigned int data_length;
    int res = wav_get_header(h_x, &format, &channels, &sr, &bits_per_sample, &data_length);
    if (!res)
    {
      cerr << "get ref header error: " << res << endl;
      std::exit(-1);
    }

    int samples = data_length * 8 / bits_per_sample;
    std::vector<int16_t> tmp(samples);
    res = wav_read_data(h_x, reinterpret_cast<unsigned char*>(tmp.data()), data_length);
    if (res < 0)
    {
      cerr << "read wav file error: " << res << endl;
      std::exit(-1);
    }
    std::vector<float> x(samples);
    std::transform(tmp.begin(), tmp.end(), x.begin(),
      [](int16_t a) {
      return static_cast<float>(a) / 32767.f;
    });

    std::cout << "Sample rate: " << sr << "Hz" << std::endl;
    return x;
    
}

int main(int argc, const char* argv[])
{
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }


  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  auto x = read_wav("/home/diaz/projects/WaveGrad/data/64346__robinhood76__00672-gunshot-1.wav");
  int sr = 16000;
  int n_fft = 1024;
  int n_hop = 300;
  std::string window = "hann";
  bool center = false;
  std::string pad_mode = "reflect";
  float power = 2.f;
  int n_mel = 80;
  int fmin = 80;
  int fmax = 8000;
  std::vector<std::vector<float>> mels = librosa::Feature::melspectrogram(x, sr, n_fft, n_hop, window, center, pad_mode, power,n_mel, fmin, fmax);

  // module.set_new_noise_schedule();
  // module->run_method("set_new_noise_schedule");
  // Copying into a tensor
  auto m = mels.size();
  auto n = mels[0].size();
  std::cout << "Size is " << m << "x" << n << std::endl;
  auto options = torch::TensorOptions().dtype(at::kFloat);
  auto tensor = torch::zeros({1, n, m}, options);
  for (int i = 0; i < n; i++)
  {
      tensor.slice(1, i,i+1) = torch::from_blob(mels[i].data(), {m}, options);
  }
  // Create a vector of inputs.
  // std::vector<torch::jit::IValue> inputs;
  // inputs.push_back(torch::ones({1, 3, 224, 224}));
  std::cout << "shape " << tensor.sizes() << std::endl;
  std::vector<torch::jit::IValue> inputs = {tensor};
  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  // std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

  std::cout << "ok\n";
}