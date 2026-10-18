#pragma once
#include "bend_registry.h"
#include <mutex>
#include <string>
#include <torch/script.h>
#include <torch/torch.h>
#include <unordered_map>
#include <vector>

class Backend {
protected:
  using ArgsType = std::vector<c10::IValue>;
  using KwargsType = std::unordered_map<std::string, c10::IValue>;
  ArgsType empty_args() { return ArgsType(); }
  KwargsType empty_kwargs() { return KwargsType(); }

  torch::jit::script::Module m_model;
  int m_loaded;
  bool m_registry_loaded = false;
  std::string m_path;
  std::mutex m_model_mutex;
  std::vector<std::string> m_available_methods;
  c10::DeviceType m_device;
  bool m_use_gpu;
  BendRegistry m_registry;

public:
  Backend();
  void perform(std::vector<float *> in_buffer, std::vector<float *> out_buffer,
               int n_vec, std::string method, int n_batches);
  bool has_method(std::string method_name);
  bool has_settable_attribute(std::string attribute);
  std::vector<std::string> get_available_methods();
  std::vector<std::string> get_available_attributes();
  std::vector<std::string> get_settable_attributes();
  std::vector<c10::IValue> get_attribute(std::string attribute_name);
  std::string get_attribute_as_string(std::string attribute_name);
  void set_attribute(std::string attribute_name,
                     std::vector<std::string> attribute_args);

  std::vector<int> get_method_params(std::string method);
  int get_higher_ratio();
  int load(std::string path);
  std::unordered_map<std::string, std::vector<int>> m_params_cache;
  int reload();
  bool is_loaded();
  torch::jit::script::Module get_model() { return m_model; }
  void use_gpu(bool value);

  std::vector<float> get_layer_weights(std::string layer_name);
  std::vector<int64_t> get_layer_shape(std::string layer_name);
  std::string get_layer_group(std::string layer_name);
  std::string get_layer_role(std::string layer_name);

  void ensure_registry_loaded();
  std::vector<std::string> get_bend_groups();
  std::vector<std::string> get_bend_layers_in_group(std::string group);
  void bend_layer(std::string layer, std::vector<float> weights);
  void reset_bend(std::string layer);
  void reset_all_bends();
};
