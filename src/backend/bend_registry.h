#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <torch/script.h>
#include <torch/torch.h>


// observed naming differences in named_parameters() for rave models
enum class RaveVariant {
    WITH_PRIOR_PREFIXED, // has _rave. prefix + encode_amortized method
    WITH_PRIOR_FLAT, // no _rave. prefix, no encode_amortized
    ADAIN, // has update_adain, no BatchNorm
    ADAIN_ALPHA, // adain + alpha parameters
    UNKNOWN
};

struct BendLayer {
    std::string name; // exact key in named_parameters()
    std::string group; // machine-readable category e.g. "encoder_bn"
    std::string role; // display string shown to the user 
};

// NOT thread-safe on its own. All public methods must be called with the
// owning Backend's m_model_mutex held. BendRegistry never acquires a lock itself.
class BendRegistry {
public:
    BendRegistry() = default;

    // ID a loaded model and populate the curated layer list.
    // is_loaded() returns false if the model variant could not be identified.
    void load(const torch::jit::script::Module &model);

    // which variant was detected
    RaveVariant variant() const { return m_variant; }

    // group names in insertion order — use these to populate an umenu
    std::vector<std::string> get_groups() const;

    // all layers belonging to a group
    std::vector<std::string> get_layers_in_group(const std::string &group) const;

    // flat list of all curated layer names
    std::vector<std::string> get_all_layers_list() const;

    // display string for a layer — what gets shown to the user
    std::string get_role(const std::string &layer) const;
    std::string get_group(const std::string &layer) const;
    std::vector<int64_t> get_layer_shape(const std::string &layer) const;

    // bending values — set a tensor offset for a layer, persists across model reloads.
    // Returns false (and logs) if the layer name is not registered; caller should throw.
    bool set_bend(const std::string &layer, torch::Tensor value);
    void reset_bend(const std::string &layer, torch::jit::script::Module &model);
    void reset_all(torch::jit::script::Module &model);
    bool has_bend(const std::string &layer) const;
    torch::Tensor get_bend(const std::string &layer) const;

    // push all stored bending values back into a freshly loaded model
    bool apply_to_model(torch::jit::script::Module &model);

    bool is_loaded() const { return m_loaded; }
    bool is_dirty() const { return m_dirty; }

private:
    RaveVariant m_variant = RaveVariant::UNKNOWN;
    bool m_loaded = false;
    bool m_dirty = false;

    std::vector<BendLayer> m_layers;
    std::vector<std::string> m_group_order;
    std::unordered_map<std::string, std::vector<std::string>> m_groups;

    // stored bending values — survive model reloads
    std::unordered_map<std::string, torch::Tensor> m_bend_state;
    std::unordered_map<std::string, torch::Tensor> m_original_state;

    static RaveVariant detect_variant(const torch::jit::script::Module &model);
    void populate(const torch::jit::script::Module &model);
    void add_layer(const std::string &name, const std::string &group, const std::string &role, const torch::Tensor &weight);
    const BendLayer *find_layer(const std::string &name) const;
};