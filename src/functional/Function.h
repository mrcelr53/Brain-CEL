/*
 * SPDX-License-Identifier: AGPL-3.0-or-later
 * Copyright (C) 2026 Marcel Rinder
 */

#ifndef FUNCTION_H
#define FUNCTION_H

#include <map>
#include <thread>
#include <string>
#include <vector>

#include <braincel/Log.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <Python.h>
#ifdef __cplusplus
}
#endif


#include "Variable.h"


template <typename InPortTypePack, typename OutPortTypePack, typename FunctionProcessTypePack = TypePack<>>
class Function final : public Module<Function<InPortTypePack, OutPortTypePack, FunctionProcessTypePack>, FunctionProcessTypePack> {
    friend class Module<Function, FunctionProcessTypePack>;

public:
    using ModuleBase = Module<Function, FunctionProcessTypePack>;

    explicit Function(Host* host) : Timed(host), ModuleBase(host) {}
    ~Function() override { cleanupPython(); }

    Function(Function&&) noexcept = default;
    Function& operator=(Function&&) noexcept = default;

    explicit Function(const Function&) = delete;
    Function& operator=(const Function&) = delete;

    void setScript(const std::string& script) { script_ = script; }

    template <typename ConType>
    size_t numReservedInputs() const { return inputVars_.template capacity<ConType>(); }
    size_t numReservedInputs() const { return inputVars_.capacity(); }
    template <typename ConType>
    void reserveInputs(size_t size) { inputVars_.template reserve<ConType>(size); }
    template <typename Type, typename... Args>
    auto& emplaceInput(Args&&... args) {   // const std::pair<uint64_t, uint64_t>& uuid, const std::string& name
        auto& input = inputVars_.template emplace<Type>(this->host(), std::forward<Args>(args)...);
        input.setClassName(Timed::className() + "(Input)");   // Class specific name
        input.setId(this->id());        // Instance specific id
        // input.setClassUuid(uuid);       // Class specific id
        // input.setName(name);            // Instance specific name
        return input;
    }
    auto& inputs() { return inputVars_; }
    auto& inputs() const { return inputVars_; }
    size_t numInputs() const { return inputVars_.size(); }
    size_t numInputConnections() const {
        size_t size = 0;
        inputVars_.forEach([&size](auto& inVar) {
            size += inVar.numOutputs();
        });
        return size;
    }

    template <typename ConType>
    size_t numReservedOutputs() const { return outputVars_.template capacity<ConType>(); }
    size_t numReservedOutputs() const { return outputVars_.capacity(); }
    template <typename ConType>
    void reserveOutputs(size_t size) { outputVars_.template reserve<ConType>(size); }
    auto& outputs() { return outputVars_; }
    auto& outputs() const { return outputVars_; }
    template <typename Type, typename... Args>
    auto& emplaceOutput(Args&&... args) {   // const std::pair<uint64_t, uint64_t>& uuid, const std::string& name
        auto& output = outputVars_.template emplace<Type>(this->host(), std::forward<Args>(args)...);
        output.setClassName(Timed::className() + "(Output)");   // Class specific name
        output.setId(this->id());   // Instance specific id
        // output.setClassUuid(uuid);  // Class specific id
        // output.setName(name);       // Instance specific name
        return output;
    }
    size_t numOutputs() const { return outputVars_.size(); }
    size_t numOutputConnections() const {
        size_t size = 0;
        outputVars_.forEach([&size](auto& outVar) {
            size += outVar.numOutputs();
        });
        return size;
    }

protected:
    bool execute() {
        if (!isSetup_) { setupPython(); }
        if (executePython()) { return true; }
        return false;
    }
    bool propagate() {
        outputVars_.forEach([](auto& outVar) {
            outVar.propagate();
        });
        return true;
    }

private:
    void setupPython() {
        if (script_.empty()) {
            BC_WARN("Function", "cannot set up Python with an empty script");
            return;
        }

        const PyGILState_STATE gstate = PyGILState_Ensure();      // Ensure GIL is set

        try {
            cleanupPython();

            const std::string moduleName = "dynamic_script_" + std::to_string(reinterpret_cast<uintptr_t>(this));
            pModule_ = PyModule_New(moduleName.c_str());
            if (!pModule_) {
                BC_ERROR("Function", "PyModule_New returned nullptr for module: {}", moduleName);
                PyErr_Print();
                PyGILState_Release(gstate);
                return;
            }

            // Get state dictionary
            PyObject* pDict = PyModule_GetDict(pModule_);
            if (!pDict) {
                BC_ERROR("Function", "failed to get module dictionary in thread: {}", std::this_thread::get_id());
                PyErr_Print();
                Py_DECREF(pModule_);
                pModule_ = nullptr;
                PyGILState_Release(gstate);
                return;
            }

            // Update simulation context
            addToPythonDict(pDict, "T", 0.);
            addToPythonDict(pDict, "DT", this->timestep());

            // Add __builtins__ for imports
            PyObject* builtins = PyEval_GetBuiltins();
            if (PyDict_SetItemString(pDict, "__builtins__", builtins) < 0) {
                BC_ERROR("Function", "failed to set __builtins__");
                PyErr_Print();
            }
            PyObject* numpy = PyImport_ImportModule("numpy");
            if (numpy) {
                PyDict_SetItemString(pDict, "numpy", numpy);
                Py_DECREF(numpy);
            } else {
                BC_WARN("Function", "failed to import numpy");
                PyErr_Print();
            }

            // Compile bytecode
            pByteCode_ = Py_CompileString(script_.c_str(), "<string>", Py_file_input);
            if (!pByteCode_) {
                BC_ERROR("Function", "failed to compile script in thread: {}", std::this_thread::get_id());
                PyErr_Print();
                Py_DECREF(pModule_);
                pModule_ = nullptr;
                PyGILState_Release(gstate);
                return;
            }

            // Execute script initially - execute __main__
            PyObject* execResult = PyEval_EvalCode(pByteCode_, pDict, pDict);
            if (!execResult) {
                PyErr_Print();
                PyGILState_Release(gstate);
                return;
            }
            Py_XDECREF(execResult);

            // Cache tick function
            pTickFunc_ = PyDict_GetItemString(pDict, "tick");
            if (pTickFunc_) { Py_INCREF(pTickFunc_); }

            // Load output keys order
            outputKeys_.clear();
            outputVars_.forEach([this](const auto& outVar) { outputKeys_.push_back(outVar.key()); });

            isSetup_ = true;

            PyGILState_Release(gstate);

        }
        catch (...) {
            PyGILState_Release(gstate);
            throw;
        }
    }
    bool executePython() {
        const PyGILState_STATE gstate = PyGILState_Ensure();      // Ensure GIL is set
        try {
            if (!pByteCode_ || !pModule_) {
                BC_ERROR("Function", "bytecode or module is null in execute() for thread: {}", std::this_thread::get_id());
                return false;
            }

            // Get state dictionary
            PyObject* pDict = PyModule_GetDict(pModule_);
            if (!pDict) {
                BC_ERROR("Function", "failed to get module dictionary in execute()");
                PyGILState_Release(gstate);
                return false;
            }

            // Update simulation context
            addToPythonDict(pDict, "T", this->currentTick());
            addToPythonDict(pDict, "DT", this->timestep());

            if (pTickFunc_) {
                // Build kwargs dict for inputs
                PyObject* kwargs = PyDict_New();
                if (!kwargs) {
                    PyGILState_Release(gstate);
                    return false;
                }

                bool inputSuccess = true;
                inputVars_.forEach([&]<typename InType>(InType& inVar) {
                    if (!inputSuccess) return;
                    PyObject* pyVal = valueToPy(inVar.withdraw());
                    const int returned = PyDict_SetItemString(kwargs, inVar.key().c_str(), pyVal);
                    if (!pyVal || returned < 0) {
                        inputSuccess = false;
                    }
                    Py_XDECREF(pyVal);
                });

                if (!inputSuccess) {
                    Py_DECREF(kwargs);
                    PyGILState_Release(gstate);
                    return false;
                }

                // Call tick
                PyObject* args = PyTuple_New(0);
                PyObject* result = PyObject_Call(pTickFunc_, args, kwargs);
                Py_DECREF(args);
                Py_DECREF(kwargs);

                if (!result) {
                    PyErr_Print();
                    PyGILState_Release(gstate);
                    return false;
                }

                // Parse return value
                if (!parseReturnValue(result)) {
                    Py_DECREF(result);
                    PyGILState_Release(gstate);
                    return false;
                }
                Py_DECREF(result);

            }
            else {
                // Fallback: no tick function - use global dict approach
                bool inputSuccess = true;
                inputVars_.forEach([&]<typename InType>(const InType& inVar) {
                    if (!inputSuccess) return;
                    inputSuccess = addToPythonDict(pDict, inVar.key(), inVar.value());
                });
                if (!inputSuccess) {
                    PyGILState_Release(gstate);
                    return false;
                }

                PyObject* execResult = PyEval_EvalCode(pByteCode_, pDict, pDict);
                if (!execResult) {
                    PyErr_Print();
                    PyGILState_Release(gstate);
                    return false;
                }
                Py_DECREF(execResult);

                // Read outputs from globals
                bool outputSuccess = true;
                outputVars_.forEach([&]<typename OutType>(OutType& outVar) {
                    if (!outputSuccess) return;
                    PyObject* pResult = PyDict_GetItemString(pDict, outVar.key().c_str());
                    if (!pResult) {
                        BC_ERROR("Function", "output '{}' not found", outVar.key());
                        outputSuccess = false;
                        return;
                    }
                    outVar.setValue(pyToValue<typename SignalType_t<OutType>::type>(pResult));
                });
                if (!outputSuccess) {
                    PyGILState_Release(gstate);
                    return false;
                }
            }

            PyGILState_Release(gstate);
            return true;
        }
        catch (...) {
            PyGILState_Release(gstate);
            throw;
        }
    }
    void cleanupPython() {
        const PyGILState_STATE gstate = PyGILState_Ensure();
        Py_CLEAR(pTickFunc_);
        Py_CLEAR(pModule_);
        Py_CLEAR(pByteCode_);
        isSetup_ = false;
        PyGILState_Release(gstate);
    }

    bool parseReturnValue(PyObject* result) {
        const size_t numOutputs = outputKeys_.size();

        if (numOutputs == 0) { return true; }               // ignore when no outputs
        if (!result || result == Py_None) { return true; }  // ignore when returned None

        // Single output
        if (numOutputs == 1) {
            if (PyDict_Check(result)) {
                PyObject* val = PyDict_GetItemString(result, outputKeys_[0].c_str());
                if (!val || val == Py_None) { return true; }  // ignore when returned None
                if (!val) {
                    BC_ERROR("Function", "key '{}' not found in returned dict", outputKeys_[0]);
                    return false;
                }
                return setOutputByIndex(0, val);
            }
            return setOutputByIndex(0, result);
        }

        // Multiple output
        if (PyDict_Check(result)) {
            // Dict format
            for (size_t i = 0; i < numOutputs; ++i) {
                PyObject* val = PyDict_GetItemString(result, outputKeys_[i].c_str());
                if (!val || val == Py_None) { continue; }  // ignore when returned None
                if (!setOutputByIndex(i, val)) return false;
            }
        }
        else if (PyTuple_Check(result)) {
            // Tuple format
            const Py_ssize_t tupleSize = PyTuple_Size(result);
            if (static_cast<size_t>(tupleSize) != numOutputs) {
                BC_ERROR("Function", "tuple size {} != expected outputs {}", tupleSize, numOutputs);
                return false;
            }
            for (size_t i = 0; i < numOutputs; ++i) {
                PyObject* val = PyTuple_GetItem(result, static_cast<Py_ssize_t>(i));  // borrowed ref
                if (!val || val == Py_None) { continue; }  // ignore when returned None
                if (!setOutputByIndex(i, val)) return false;
            }
        }
        else {
            BC_ERROR("Function", "expected dict or tuple for multiple outputs");
            return false;
        }

        return true;
    }
    bool setOutputByIndex(const size_t index, PyObject* pyVal) {
        if (!pyVal || pyVal == Py_None) { return true; }  // ignore if None

        size_t currentIdx = 0;
        outputVars_.forEach([&]<typename OutType>(OutType& outVar) {
            if (currentIdx == index) {
                using SignalType = SignalType_t<OutType>::type;
                outVar.setValue(pyToValue<SignalType>(pyVal));
            }
            ++currentIdx;
        });
        return true;
    }

    template <typename ValueType>
    bool addToPythonDict(PyObject* pyDict, const std::string& key, ValueType value) {
        PyObject* pyValue = valueToPy<ValueType>(value);
        if (!pyValue) {
            BC_ERROR("Function", "failed to convert value to PyObject for key: {}", key);
            return false;
        }
        if (PyDict_SetItemString(pyDict, key.c_str(), pyValue) < 0) {
            BC_ERROR("Function", "failed to set global variable for key: {}", key);
            Py_DECREF(pyValue);
            return false;
        }
        Py_DECREF(pyValue);
        return true;
    }

    template <typename T>
    static PyObject* valueToPy(const T& val) {
        using U = std::decay_t<T>;

        if constexpr (std::is_same_v<U, bool>) {
            return PyBool_FromLong(val ? 1 : 0);
        }
        else if constexpr (std::is_integral_v<U> && std::is_signed_v<U>) {
            return PyLong_FromLongLong(static_cast<long long>(val));
        }
        else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U>) {
            return PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(val));
        }
        else if constexpr (std::is_floating_point_v<U>) {
            return PyFloat_FromDouble(static_cast<double>(val));
        }
        else if constexpr (std::is_same_v<U, std::string>) {
            return PyUnicode_FromString(val.c_str());
        }
        else if constexpr (requires {
            { val.size() } -> std::convertible_to<size_t>;
            { val[std::declval<size_t>()] } -> std::convertible_to<typename U::value_type>;
            typename U::value_type;
        } && std::is_same_v<U, std::vector<typename U::value_type>>) {
            using Elem = U::value_type;
            PyObject* list = PyList_New(static_cast<Py_ssize_t>(val.size()));
            if (!list) return nullptr;

            bool success = true;
            for (size_t i = 0; i < val.size(); ++i) {
                PyObject* item = nullptr;
                if constexpr (std::is_same_v<std::decay_t<Elem>, bool>) {
                    item = PyBool_FromLong(val[i] ? 1 : 0);
                } else if constexpr (std::is_integral_v<Elem> && std::is_signed_v<Elem>) {
                    item = PyLong_FromLongLong(static_cast<long long>(val[i]));
                } else if constexpr (std::is_integral_v<Elem> && std::is_unsigned_v<Elem>) {
                    item = PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(val[i]));
                } else if constexpr (std::is_floating_point_v<Elem>) {
                    item = PyFloat_FromDouble(static_cast<double>(val[i]));
                } else if constexpr (std::is_same_v<std::decay_t<Elem>, std::string>) {
                    item = PyUnicode_FromString(val[i].c_str());
                } else {
                    success = false;
                    break;
                }
                if (!item || PyList_SetItem(list, static_cast<Py_ssize_t>(i), item) < 0) {
                    success = false;
                    Py_XDECREF(item);
                    break;
                }
                // PyList_SetItem steals ref. No need to decref item
            }

            if (!success) {
                Py_DECREF(list);
                BC_ERROR("Function", "failed to convert vector element in valueToPy");
                Py_INCREF(Py_None);
                return Py_None;
            }
            return list;
        } else {
            BC_ERROR("Function", "unsupported type in valueToPy");
            Py_INCREF(Py_None);
            return Py_None;
        }
    }

    template <typename T>
    static T pyToValue(PyObject* obj) {
        using U = std::decay_t<T>;

        if (!obj || obj == Py_None) { return U{}; }  // ignore None

        if constexpr (std::is_same_v<U, bool>) {
            if (PyBool_Check(obj)) {
                return obj == Py_True;
            }
            BC_WARN("Function", "expected bool - defaulting to false");
            return false;
        }
        else if constexpr (std::is_integral_v<U> && std::is_signed_v<U>) {
            long long ll = 0;
            bool valid = false;
            if (PyLong_Check(obj)) {
                ll = PyLong_AsLongLong(obj);
                if (!PyErr_Occurred()) valid = true;
                PyErr_Clear();
            }
            else if (PyFloat_Check(obj)) {
                double d = PyFloat_AsDouble(obj);
                if (!PyErr_Occurred()) {
                    long long ll_cand = static_cast<long long>(d);
                    if (d == static_cast<double>(ll_cand)) {
                        ll = ll_cand;
                        valid = true;
                    }
                }
                PyErr_Clear();
            }
            if (valid) {
                if (ll < std::numeric_limits<U>::min() || ll > std::numeric_limits<U>::max()) {
                    BC_WARN("Function", "value out of range for {}", typeid(U).name());
                    return U{0};
                }
                return static_cast<U>(ll);
            }
            BC_WARN("Function", "expected number for signed integer - defaulting to 0");
            return U{0};
        }
        else if constexpr (std::is_integral_v<U> && std::is_unsigned_v<U>) {
            unsigned long long ull = 0;
            bool valid = false;
            if (PyLong_Check(obj)) {
                ull = PyLong_AsUnsignedLongLong(obj);
                if (!PyErr_Occurred()) valid = true;
                PyErr_Clear();
            }
            else if (PyFloat_Check(obj)) {
                double d = PyFloat_AsDouble(obj);
                if (!PyErr_Occurred() && d >= 0.0) {
                    unsigned long long ull_cand = static_cast<unsigned long long>(d);
                    if (d == static_cast<double>(ull_cand)) {
                        ull = ull_cand;
                        valid = true;
                    }
                }
                PyErr_Clear();
            }
            if (valid) {
                if (ull > std::numeric_limits<U>::max()) {
                    BC_WARN("Function", "value out of range for {}", typeid(U).name());
                    return U{0};
                }
                return static_cast<U>(ull);
            }
            BC_WARN("Function", "expected non-negative number for unsigned integer - defaulting to 0");
            return U{0};
        }
        else if constexpr (std::is_floating_point_v<U>) {
            double d = 0.0;
            bool valid = false;
            if (PyFloat_Check(obj)) {
                d = PyFloat_AsDouble(obj);
                if (!PyErr_Occurred()) valid = true;
                PyErr_Clear();
            }
            else if (PyLong_Check(obj)) {
                const long long ll = PyLong_AsLongLong(obj);
                if (!PyErr_Occurred()) {
                    d = static_cast<double>(ll);
                    valid = true;
                }
                PyErr_Clear();
            }
            if (!valid) {
                BC_WARN("Function", "expected number for float - defaulting to 0.0");
            }
            return static_cast<U>(d);
        }
        else if constexpr (std::is_same_v<U, std::string>) {
            if (PyUnicode_Check(obj)) {
                Py_ssize_t size;
                const char* str = PyUnicode_AsUTF8AndSize(obj, &size);
                if (str) {
                    return std::string(str, static_cast<size_t>(size));
                }
            }
            BC_WARN("Function", "expected string - defaulting to empty");
            return std::string{};
        }

        // Vectors
        else if constexpr ( requires(U u) {
            { u.size() } -> std::convertible_to<size_t>;
            { u[std::declval<size_t>()] } -> std::convertible_to<typename U::value_type>;
            typename U::value_type; } && std::is_same_v<U, std::vector<typename U::value_type>>) {

            using V = typename U::value_type;

            // Clear any existing error before buffer check
            PyErr_Clear();

            // Try buffer for arrays
            if (PyObject_CheckBuffer(obj)) {
                auto result = pyBufferToVector<V>(obj);
                if (!result.empty() || PyErr_Occurred()) {
                    PyErr_Clear();
                    if (!result.empty()) {
                        return result;
                    }
                }
            }

            // Fallback to list/tuple
            if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
                BC_WARN("Function", "expected list or tuple for vector - defaulting to empty");
                return U{};
            }

            const Py_ssize_t len = PySequence_Size(obj);
            if (len < 0) {
                BC_WARN("Function", "failed to get sequence size - defaulting to empty vector");
                PyErr_Clear();
                return U{};
            }

            U res;
            res.reserve(static_cast<size_t>(len));

            for (Py_ssize_t i = 0; i < len; ++i) {
                PyObject* item = PySequence_GetItem(obj, i);  // new reference
                if (!item) {
                    BC_WARN("Function", "failed to get sequence item at index {}", i);
                    PyErr_Clear();
                    res.push_back(V{});
                    continue;
                }
                V conv = pyToValue<V>(item);
                Py_DECREF(item);  // release reference!!
                res.push_back(conv);
            }

            return res;
        }

        else {
            BC_ERROR("Function", "unsupported type {} in pyToValue", typeid(U).name());
            return U{};
        }
    }
    template <typename V>
    static std::vector<V> pyBufferToVector(PyObject* obj) {
        Py_buffer view;
        if (PyObject_GetBuffer(obj, &view, PyBUF_CONTIG_RO | PyBUF_FORMAT) != 0) {
            PyErr_Clear();
            return std::vector<V>{};
        }

        std::vector<V> res;

        if (view.ndim != 1 || view.itemsize <= 0) {
            PyBuffer_Release(&view);
            return res;
        }

        size_t len = static_cast<size_t>(view.len) / view.itemsize;
        res.reserve(len);

        bool success = false;

        // Handle double vectors
        if constexpr (std::is_same_v<V, double>) {
            if (strcmp(view.format, "d") == 0) {
                const auto* ptr = static_cast<const double*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
            else if (strcmp(view.format, "f") == 0) {
                const auto* ptr = static_cast<const float*>(view.buf);
                for (size_t i = 0; i < len; ++i) {
                    res.push_back(static_cast<double>(ptr[i]));
                }
                success = true;
            }
        }
        // Handle float vectors
        else if constexpr (std::is_same_v<V, float>) {
            if (strcmp(view.format, "f") == 0) {
                const auto* ptr = static_cast<const float*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
            else if (strcmp(view.format, "d") == 0) {
                const auto* ptr = static_cast<const double*>(view.buf);
                for (size_t i = 0; i < len; ++i) {
                    res.push_back(static_cast<float>(ptr[i]));
                }
                success = true;
            }
        }
        // Handle int8_t vectors
        else if constexpr (std::is_same_v<V, int8_t>) {
            if (strcmp(view.format, "b") == 0) {
                const auto* ptr = static_cast<const int8_t*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
        }
        // Handle uint8_t vectors
        else if constexpr (std::is_same_v<V, uint8_t>) {
            if (strcmp(view.format, "B") == 0) {
                const auto* ptr = static_cast<const uint8_t*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
        }
        // Handle int16_t vectors
        else if constexpr (std::is_same_v<V, int16_t>) {
            if (strcmp(view.format, "h") == 0) {
                const auto* ptr = static_cast<const int16_t*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
        }
        // Handle uint16_t vectors
        else if constexpr (std::is_same_v<V, uint16_t>) {
            if (strcmp(view.format, "H") == 0) {
                const auto* ptr = static_cast<const uint16_t*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
        }
        // Handle int32_t vectors
        else if constexpr (std::is_same_v<V, int32_t>) {
            if (strcmp(view.format, "i") == 0 || strcmp(view.format, "l") == 0) {
                const auto* ptr = static_cast<const int32_t*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
        }
        // Handle uint32_t vectors
        else if constexpr (std::is_same_v<V, uint32_t>) {
            if (strcmp(view.format, "I") == 0 || strcmp(view.format, "L") == 0) {
                const auto* ptr = static_cast<const uint32_t*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
        }
        // Handle int64_t vectors
        else if constexpr (std::is_same_v<V, int64_t>) {
            if (strcmp(view.format, "q") == 0 || strcmp(view.format, "l") == 0) {
                const auto* ptr = static_cast<const int64_t*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
        }
        // Handle uint64_t vectors
        else if constexpr (std::is_same_v<V, uint64_t>) {
            if (strcmp(view.format, "Q") == 0 || strcmp(view.format, "L") == 0) {
                const auto* ptr = static_cast<const uint64_t*>(view.buf);
                res.assign(ptr, ptr + len);
                success = true;
            }
        }

        PyBuffer_Release(&view);

        if (!success) {
            res.clear();
        }

        return res;
    }

    MixedMultiStorage<InPortTypePack> inputVars_;
    MixedMultiStorage<OutPortTypePack> outputVars_;
    std::vector<std::string> outputKeys_;

    std::string script_;
    bool isSetup_ = false;

    std::string scriptPath_;
    PyObject* pModule_ = nullptr;
    PyObject* pByteCode_ = nullptr;
    PyObject* pTickFunc_ = nullptr;
};


/// Copy-constructible specialization. This ensures the hash is invocable and copyable
namespace std {
    template <typename Pre, typename Post, typename Proc>
    struct hash<Function<Pre, Post, Proc>> {
        size_t operator()(const Function<Pre, Post, Proc>& obj) const noexcept {
            return std::hash<const void*>{}(static_cast<const void*>(&obj));
        }
    };
}

#endif // FUNCTION_H