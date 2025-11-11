#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Program.h"
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

using namespace llvm;

// Command line options
static cl::opt<std::string> InputFilename(cl::Positional, cl::desc("<input bitcode file>"), cl::Required);
static cl::opt<std::string> OutputFilename(cl::Positional, cl::desc("<output bitcode file>"), cl::Required);
static cl::opt<bool> UseAI("use-ai", cl::desc("Use AI-driven optimization"), cl::init(false));
static cl::opt<std::string> ModelPath("model-path", cl::desc("Path to trained AI model"), cl::init("models/best_model.pt"));
static cl::opt<int> OptLevel("O", cl::desc("Optimization level (0-3)"), cl::init(2));

// Function to run AI optimizer via Python bridge
bool runAIOptimizer(const std::string& inputFile, const std::string& outputFile, const std::string& modelPath) {
    // Create a temporary file for intermediate IR
    int tempFD;
    SmallString<128> tempFilename;
    std::error_code EC = sys::fs::createTemporaryFile("ai_opt", "ll", tempFD, tempFilename);
    if (EC) {
        errs() << "Error creating temporary file: " << EC.message() << "\n";
        return false;
    }
    raw_fd_ostream tempOut(tempFD, true);
    tempOut.close();
    
    // Construct command to run Python AI optimizer
    std::string pythonCmd = "python ai_optimizer_bridge.py";
    pythonCmd += " --input-ir=\"" + inputFile + "\"";
    pythonCmd += " --output-ir=\"" + outputFile + "\"";
    pythonCmd += " --model-path=\"" + modelPath + "\"";
    
    // Run the Python script
    errs() << "Running AI optimizer: " << pythonCmd << "\n";
    int result = std::system(pythonCmd.c_str());
    
    return (result == 0);
}

// Function to read pass sequence from file
std::vector<std::string> readPassSequence(const std::string& filename) {
    std::vector<std::string> passes;
    std::ifstream file(filename);
    std::string line;
    
    if (file.is_open()) {
        while (std::getline(file, line)) {
            if (!line.empty()) {
                passes.push_back(line);
            }
        }
        file.close();
    }
    
    return passes;
}

// Function to apply a specific sequence of passes
void applyPassSequence(Module &M, const std::vector<std::string>& passes) {
    // Create PassBuilder and Analysis Managers
    PassBuilder PB;
    FunctionAnalysisManager FAM;
    LoopAnalysisManager LAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;
    
    // Register analyses with PassBuilder
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerModuleAnalyses(MAM);
    
    // Cross register proxies between managers
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
    
    // Create module pass manager
    ModulePassManager MPM;
    
    // Add passes based on the sequence
    for (const auto& pass : passes) {
        errs() << "Applying pass: " << pass << "\n";
        
        // Add the appropriate pass based on name
        // This is a simplified version - in reality, you'd have a more complete mapping
        if (pass == "mem2reg") {
            MPM.addPass(createModuleToFunctionPassAdaptor(PromotePass()));
        } else if (pass == "sroa") {
            MPM.addPass(createModuleToFunctionPassAdaptor(SROA()));
        } else if (pass == "early-cse") {
            MPM.addPass(createModuleToFunctionPassAdaptor(EarlyCSEPass()));
        } else if (pass == "simplifycfg") {
            MPM.addPass(createModuleToFunctionPassAdaptor(SimplifyCFGPass()));
        } else if (pass == "gvn") {
            MPM.addPass(createModuleToFunctionPassAdaptor(GVNPass()));
        } else if (pass == "instcombine") {
            MPM.addPass(createModuleToFunctionPassAdaptor(InstCombinePass()));
        } else if (pass == "inline") {
            MPM.addPass(createModuleToFunctionPassAdaptor(InlinerPass()));
        } else if (pass == "loop-unroll") {
            MPM.addPass(createModuleToFunctionPassAdaptor(LoopUnrollPass()));
        } else {
            errs() << "Warning: Unsupported pass '" << pass << "', skipping\n";
        }
    }
    
    // Run the pass manager
    MPM.run(M, MAM);
}

int main(int argc, char** argv) {
    // Parse command line options
    cl::ParseCommandLineOptions(argc, argv, "LLVM AI-driven Optimizer\n");

    // Initialize LLVM targets
    InitializeAllTargets();
    InitializeAllTargetMCs();
    InitializeAllAsmPrinters();
    InitializeAllAsmParsers();

    // Create LLVM context and load module
    LLVMContext context;
    SMDiagnostic err;
    std::unique_ptr<Module> module = parseIRFile(InputFilename, err, context);
    if (!module) {
        errs() << "Error reading IR file\n";
        err.print(argv[0], errs());
        return 1;
    }

    // Set target triple to host
    std::string targetTriple = sys::getProcessTriple();
    module->setTargetTriple(targetTriple);

    // Check if we should use AI-driven optimization
    if (UseAI) {
        errs() << "Using AI-driven optimization\n";
        
        // Save the module to a temporary file
        std::error_code EC;
        raw_fd_ostream tempOut("temp_input.ll", EC, sys::fs::OF_None);
        if (EC) {
            errs() << "Could not open temporary file\n";
            return 1;
        }
        module->print(tempOut, nullptr);
        tempOut.close();
        
        // Run AI optimizer
        bool success = runAIOptimizer("temp_input.ll", OutputFilename, ModelPath);
        
        // Clean up temporary file
        sys::fs::remove("temp_input.ll");
        
        if (success) {
            errs() << "AI optimization completed successfully.\n";
            return 0;
        } else {
            errs() << "AI optimization failed. Falling back to standard optimization.\n";
            // Fall through to standard optimization
        }
    }

    // Standard optimization pipeline
    PassBuilder PB;
    FunctionAnalysisManager FAM;
    LoopAnalysisManager LAM;
    CGSCCAnalysisManager CGAM;
    ModuleAnalysisManager MAM;

    // Register analyses with PassBuilder
    PB.registerFunctionAnalyses(FAM);
    PB.registerLoopAnalyses(LAM);
    PB.registerCGSCCAnalyses(CGAM);
    PB.registerModuleAnalyses(MAM);

    // Cross register proxies between managers
    PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

    // Build the optimization pipeline based on opt level
    OptimizationLevel OL;
    switch (OptLevel) {
        case 0: OL = OptimizationLevel::O0; break;
        case 1: OL = OptimizationLevel::O1; break;
        case 3: OL = OptimizationLevel::O3; break;
        default: OL = OptimizationLevel::O2; break;
    }
    
    ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(OL);

    // Run the optimization pipeline on the module
    MPM.run(*module, MAM);

    // Write optimized module to output file
    std::error_code EC;
    raw_fd_ostream outFile(OutputFilename, EC, sys::fs::OF_None);
    if (EC) {
        errs() << "Could not open file: " << OutputFilename << "\n";
        return 1;
    }

    module->print(outFile, nullptr);
    errs() << "Standard optimization completed successfully.\n";

    return 0;
}
