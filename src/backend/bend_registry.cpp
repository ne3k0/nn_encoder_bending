#include "bend_registry.h"
#include <cstdio>
#include <sstream>
#include <unordered_map>

// reads model to determine which naming paradigm the imported model follows
RaveVariant BendRegistry::detect_variant(const torch::jit::script::Module &model) {
    bool has_update_adain = false;
    bool has_encode_amortized = false;
    bool has_rave_prefix = false;
    bool has_encoder_alpha = false;
    bool has_encoder = false;

    for (const auto &method : model.get_methods()) {
        const std::string &mn = method.name();
        if (mn == "update_adain") { has_update_adain = true; }
        if (mn == "encode_amortized") { has_encode_amortized = true; }
    }

    for (const auto &buf : model.named_buffers()) {
        if (buf.name.find("_rave.encoder") != std::string::npos) {
            has_rave_prefix = true;
        }
    }

    for (const auto &param : model.named_parameters()) {
        if (param.name.find("encoder") != std::string::npos) {
            has_encoder = true;
        }
        // restrict alpha check to encoder scope — some models have decoder.*.alpha too
        if (param.name.find("encoder.encoder.net.") != std::string::npos &&
            param.name.find(".alpha") != std::string::npos) {
            has_encoder_alpha = true;
        }
    }

    if (!has_encoder && !has_update_adain) {
        return RaveVariant::UNKNOWN;
    }

    if (has_update_adain) {
        if (has_encoder_alpha) {
            return RaveVariant::ADAIN_ALPHA;
        } else {
            return RaveVariant::ADAIN;
        }
    }

    // encode_amortized is unique to with-prior models; buffer prefix as fallback for older exports
    if (has_encode_amortized || has_rave_prefix) {
        return RaveVariant::WITH_PRIOR_PREFIXED;
    }

    return RaveVariant::WITH_PRIOR_FLAT;
}


void BendRegistry::load(const torch::jit::script::Module &model) {
    RaveVariant new_variant = detect_variant(model);
    if (new_variant != m_variant) {
        m_bend_state.clear();
    }
    m_variant = new_variant;
    m_layers.clear();
    m_groups.clear();
    m_group_order.clear();
    m_original_state.clear();
    populate(model);
    m_loaded = true;
}


void BendRegistry::add_layer(const std::string &name, const std::string &group, const std::string &role, const torch::Tensor &weight) {
    m_layers.push_back({name, group, role});
    if (m_groups.find(group) == m_groups.end()) {
        m_group_order.push_back(group);
    }
    m_groups[group].push_back(name);
    m_original_state[name] = weight.clone();
}


void BendRegistry::populate(const torch::jit::script::Module &model) {
    if (m_variant == RaveVariant::UNKNOWN) {
        return;
    }

    // single pass to build a lookup map — used for bias existence checks and original weight capture
    std::unordered_map<std::string, torch::Tensor> param_map;
    for (const auto &param : model.named_parameters()) {
        param_map[param.name] = param.value;
    }

    struct ConvInfo { std::string name; int64_t out_ch; };

    auto split_latent = [&](std::vector<ConvInfo> &convs,
                            std::string &latent_name,
                            std::string &latent_bias_name) {
        if (convs.empty()) { return; }
        // the RAVE latent projection always reduces channel count vs the preceding conv
        if (convs.size() >= 2 && convs.back().out_ch > convs[convs.size() - 2].out_ch) {
            fprintf(stderr, "BendRegistry: latent heuristic warning for '%s' — output channels do not decrease; check model structure\n",
                convs.back().name.c_str());
        }
        latent_name = convs.back().name;
        convs.pop_back();
        std::string lb = latent_name.substr(0, latent_name.size() - 6) + "bias";
        if (param_map.count(lb)) {
            latent_bias_name = lb;
        }
    };

    if (m_variant == RaveVariant::WITH_PRIOR_PREFIXED || m_variant == RaveVariant::WITH_PRIOR_FLAT) {
        std::string prefix;
        if (m_variant == RaveVariant::WITH_PRIOR_PREFIXED) {
            prefix = "_rave.encoder.net.";
        } else {
            prefix = "encoder.encoder.net.";
        }

        std::vector<std::string> bn_names;
        std::vector<ConvInfo> convs;

        for (const auto &param : model.named_parameters()) {
            const std::string &n = param.name;
            if (n.find(prefix) != 0) { continue; }
            // only direct net.{N}.weight — skip submodule paths (e.g. .aligned.branches)
            std::string suffix = n.substr(prefix.size());
            auto dot = suffix.find('.');
            if (dot == std::string::npos) { continue; }
            if (suffix.substr(dot + 1) != "weight") { continue; }
            if (param.value.dim() == 1) {
                bn_names.push_back(n);
            } else if (param.value.dim() == 3) {
                convs.push_back({n, param.value.sizes()[0]});
            }
        }

        std::string latent_name, latent_bias_name;

        if (m_variant == RaveVariant::WITH_PRIOR_PREFIXED) {
            // with-prior models store _rave.latent_size; latent conv outputs 2*latent_size (mean+logvar)
            int64_t latent_out_ch = -1;
            for (const auto &buf : model.named_buffers()) {
                if (buf.name == "_rave.latent_size") {
                    latent_out_ch = 2 * buf.value.item<int64_t>();
                    break;
                }
            }
            if (latent_out_ch > 0) {
                for (int i = (int)convs.size() - 1; i >= 0; --i) {
                    if (convs[i].out_ch == latent_out_ch) {
                        latent_name = convs[i].name;
                        convs.erase(convs.begin() + i);
                        std::string lb = latent_name.substr(0, latent_name.size() - 6) + "bias";
                        if (param_map.count(lb)) { latent_bias_name = lb; }
                        break;
                    }
                }
            }
            if (latent_name.empty()) {
                split_latent(convs, latent_name, latent_bias_name);
            }
        } else {
            split_latent(convs, latent_name, latent_bias_name);
        }

        int stage = 1;
        for (const auto &name : bn_names) {
            add_layer(name, "encoder_bn", "encoder stage " + std::to_string(stage) + " — channel gain", param_map.at(name));
            std::string bias = name.substr(0, name.size() - 6) + "bias";
            // quantised models may zero bn weights without removing them — bending will restore them
            if (param_map.count(bias)) {
                add_layer(bias, "encoder_bn", "encoder stage " + std::to_string(stage) + " — channel offset", param_map.at(bias));
            }
            stage++;
        }

        stage = 1;
        for (const auto &ci : convs) {
            add_layer(ci.name, "encoder_conv", "encoder stage " + std::to_string(stage) + " — conv filters", param_map.at(ci.name));
            stage++;
        }

        if (!latent_name.empty()) {
            add_layer(latent_name, "encoder_latent", "latent projection", param_map.at(latent_name));
            if (!latent_bias_name.empty()) {
                add_layer(latent_bias_name, "encoder_latent", "latent projection offset", param_map.at(latent_bias_name));
            }
        }

    } else if (m_variant == RaveVariant::ADAIN) {
        const std::string adain_prefix = "encoder.encoder.net.";
        std::vector<ConvInfo> convs;

        for (const auto &param : model.named_parameters()) {
            const std::string &n = param.name;
            if (n.find(adain_prefix) != 0) { continue; }
            // only direct net.{N}.weight — skip submodule paths
            std::string suffix = n.substr(adain_prefix.size());
            auto dot = suffix.find('.');
            if (dot == std::string::npos) { continue; }
            if (suffix.substr(dot + 1) != "weight") { continue; }
            if (param.value.dim() != 3) { continue; }
            convs.push_back({n, param.value.sizes()[0]});
        }

        std::string latent_name, latent_bias_name;
        split_latent(convs, latent_name, latent_bias_name);

        int stage = 1;
        for (const auto &ci : convs) {
            add_layer(ci.name, "encoder_conv", "encoder stage " + std::to_string(stage) + " — conv filters", param_map.at(ci.name));
            stage++;
        }

        if (!latent_name.empty()) {
            add_layer(latent_name, "encoder_latent", "latent projection", param_map.at(latent_name));
            if (!latent_bias_name.empty()) {
                add_layer(latent_bias_name, "encoder_latent", "latent projection offset", param_map.at(latent_bias_name));
            }
        }

    } else if (m_variant == RaveVariant::ADAIN_ALPHA) {
        const std::string adain_prefix = "encoder.encoder.net.";
        int stage = 1;
        for (const auto &param : model.named_parameters()) {
            const std::string &n = param.name;
            if (n.find(adain_prefix) != 0) { continue; }
            // only direct net.{N}.alpha — skip submodule paths
            std::string suffix = n.substr(adain_prefix.size());
            auto dot = suffix.find('.');
            if (dot == std::string::npos) { continue; }
            if (suffix.substr(dot + 1) != "alpha") { continue; }
            add_layer(n, "encoder_alpha", "encoder stage " + std::to_string(stage) + " — channel scale", param.value);
            stage++;
        }
    }
}

std::vector<std::string> BendRegistry::get_groups() const {
    return m_group_order;
}

std::vector<std::string> BendRegistry::get_layers_in_group(const std::string &group) const {
    auto it = m_groups.find(group);
    if (it == m_groups.end()) {
        return {};
    }
    return it->second;
}

std::vector<std::string> BendRegistry::get_all_layers_list() const {
    std::vector<std::string> names;
    names.reserve(m_layers.size());
    for (const auto &layer : m_layers) {
        names.push_back(layer.name);
    }
    return names;
}

std::string BendRegistry::get_role(const std::string &layer) const {
    for (const auto &l : m_layers) {
        if (l.name == layer) { return l.role; }
    }
    return "";
}

std::string BendRegistry::get_group(const std::string &layer) const {
    for (const auto &l : m_layers) {
        if (l.name == layer) { return l.group; }
    }
    return "";
}

std::vector<int64_t> BendRegistry::get_layer_shape(const std::string &layer) const {
    auto it = m_original_state.find(layer);
    if (it == m_original_state.end()) { return {}; }
    std::vector<int64_t> shape;
    for (auto s : it->second.sizes()) { shape.push_back(s); }
    return shape;
}

void BendRegistry::set_bend(const std::string &layer, torch::Tensor value) {
    bool found = false;
    for (const auto &l : m_layers) {
        if (l.name == layer) {
            found = true;
            break;
        }
    }
    if (!found) {
        fprintf(stderr, "BendRegistry::set_bend: unknown layer '%s'\n", layer.c_str());
        return;
    }
    auto orig_it = m_original_state.find(layer);
    if (orig_it != m_original_state.end() && value.numel() == orig_it->second.numel()) {
        m_bend_state[layer] = value.reshape(orig_it->second.sizes()).clone();
    } else {
        m_bend_state[layer] = value.clone();
    }
}

void BendRegistry::reset_bend(const std::string &layer, torch::jit::script::Module &model) {
    m_bend_state.erase(layer);
    for (auto param : model.named_parameters()) {
        if (param.name == layer) {
            auto it = m_original_state.find(layer);
            if (it != m_original_state.end()) {
                param.value.data().copy_(it->second);
            }
            break;
        }
    }
}

void BendRegistry::reset_all(torch::jit::script::Module &model) {
    m_bend_state.clear();
    for (auto param : model.named_parameters()) {
        auto it = m_original_state.find(param.name);
        if (it != m_original_state.end()) {
            param.value.data().copy_(it->second);
        }
    }
}

bool BendRegistry::has_bend(const std::string &layer) const {
    return m_bend_state.find(layer) != m_bend_state.end();
}

torch::Tensor BendRegistry::get_bend(const std::string &layer) const {
    auto it = m_bend_state.find(layer);
    if (it == m_bend_state.end()) {
        throw std::runtime_error("BendRegistry: no bend stored for layer: " + layer);
    }
    return it->second.clone();
}

void BendRegistry::apply_to_model(torch::jit::script::Module &model) const {
    for (const auto &kv : m_bend_state) {
        bool found = false;
        for (auto param : model.named_parameters()) {
            if (param.name == kv.first) {
                found = true;
                if (param.value.sizes() != kv.second.sizes()) {
                    std::ostringstream oss;
                    oss << "BendRegistry::apply_to_model: shape mismatch for '" << kv.first
                        << "' — expected " << param.value.sizes()
                        << ", got " << kv.second.sizes() << "\n";
                    fprintf(stderr, "%s", oss.str().c_str());
                    break;
                }
                param.value.data().copy_(kv.second);
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "BendRegistry::apply_to_model: layer '%s' not found in model\n", kv.first.c_str());
        }
    }
}
