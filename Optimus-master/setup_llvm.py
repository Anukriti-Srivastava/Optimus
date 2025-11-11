import os
import sys
import subprocess
import platform
import urllib.request
import zipfile
import shutil
import tempfile
import ctypes
from pathlib import Path

# LLVM version to download
LLVM_VERSION = "14.0.0"

def is_admin():
    """Check if the script is running with administrator privileges (Windows only)"""
    if platform.system() != "Windows":
        return False
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except:
        return False

def download_file(url, destination):
    """Download a file from URL to destination with progress reporting"""
    print(f"Downloading {url}...")
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        if total_size > 0:
            sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded} / {total_size} bytes)")
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, reporthook=report_progress)
        print("\nDownload completed!")
        return True
    except Exception as e:
        print(f"\nError downloading file: {e}")
        return False

def download_and_install_llvm():
    """Download and install LLVM for Windows"""
    # Determine system architecture
    is_64bit = platform.architecture()[0] == '64bit'
    
    # LLVM download URL
    if is_64bit:
        llvm_url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/LLVM-{LLVM_VERSION}-win64.exe"
    else:
        llvm_url = f"https://github.com/llvm/llvm-project/releases/download/llvmorg-{LLVM_VERSION}/LLVM-{LLVM_VERSION}-win32.exe"
    
    # Create temp directory for download
    temp_dir = tempfile.mkdtemp()
    installer_path = os.path.join(temp_dir, f"LLVM-{LLVM_VERSION}-win{'64' if is_64bit else '32'}.exe")
    
    # Download LLVM installer
    if not download_file(llvm_url, installer_path):
        print("Failed to download LLVM installer.")
        return False
    
    # Install LLVM
    print("\nInstalling LLVM...")
    
    # Default installation path
    install_path = os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'LLVM')
    
    # Run installer silently
    try:
        # If admin, install for all users, otherwise install for current user
        if is_admin():
            cmd = [installer_path, '/S', '/D=' + install_path]
        else:
            # Install to user directory if not admin
            install_path = os.path.join(os.path.expanduser('~'), 'LLVM')
            cmd = [installer_path, '/S', '/D=' + install_path]
        
        subprocess.run(cmd, check=True)
        print(f"LLVM installed to {install_path}")
        
        # Clean up
        shutil.rmtree(temp_dir)
        
        return install_path
    except Exception as e:
        print(f"Error installing LLVM: {e}")
        return False

def setup_project_llvm():
    """Set up LLVM for the project by creating a local copy of necessary files"""
    # First check if LLVM is already installed
    try:
        result = subprocess.run(['clang', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("LLVM is already installed and available in PATH.")
            return True
    except:
        print("LLVM not found in PATH. Proceeding with installation...")
    
    # Download and install LLVM
    install_path = download_and_install_llvm()
    if not install_path:
        print("Failed to install LLVM.")
        return False
    
    # Add LLVM to PATH for this session
    bin_path = os.path.join(install_path, 'bin')
    os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
    
    # Create a batch file to set PATH for future runs
    batch_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'set_llvm_path.bat')
    with open(batch_file, 'w') as f:
        f.write(f'@echo off\necho Setting LLVM path...\nset PATH={bin_path};%PATH%\n')
    
    print(f"Created {batch_file} to set LLVM path for future runs.")
    print("Run this batch file before running your Python scripts.")
    
    # Verify installation
    try:
        result = subprocess.run(['clang', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            print("\nLLVM installation verified!")
            print(result.stdout.decode())
            return True
        else:
            print("LLVM installation could not be verified.")
            return False
    except:
        print("LLVM installation could not be verified.")
        return False

def create_llvm_wrapper():
    """Create Python wrapper for LLVM tools"""
    wrapper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'llvm_wrapper')
    os.makedirs(wrapper_dir, exist_ok=True)
    
    # Create __init__.py
    with open(os.path.join(wrapper_dir, '__init__.py'), 'w') as f:
        f.write('# LLVM Wrapper Package\n')
    
    # Create llvm_tools.py
    with open(os.path.join(wrapper_dir, 'llvm_tools.py'), 'w') as f:
        f.write('''import os
import subprocess
import tempfile
import platform
from pathlib import Path

class LLVMTools:
    """Wrapper for LLVM tools"""
    
    @staticmethod
    def find_llvm_bin():
        """Find LLVM bin directory"""
        # Check if LLVM is in PATH
        try:
            if platform.system() == "Windows":
                result = subprocess.run(['where', 'clang'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                result = subprocess.run(['which', 'clang'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
            if result.returncode == 0:
                # Extract directory from path
                clang_path = result.stdout.decode().strip().split('\\n')[0]
                return str(Path(clang_path).parent)
        except:
            pass
        
        # Check common installation locations
        if platform.system() == "Windows":
            common_paths = [
                os.path.join(os.environ.get('ProgramFiles', 'C:\\Program Files'), 'LLVM', 'bin'),
                os.path.join(os.environ.get('ProgramFiles(x86)', 'C:\\Program Files (x86)'), 'LLVM', 'bin'),
                os.path.join(os.path.expanduser('~'), 'LLVM', 'bin')
            ]
        else:
            common_paths = [
                '/usr/bin',
                '/usr/local/bin',
                '/opt/llvm/bin'
            ]
        
        for path in common_paths:
            if os.path.exists(os.path.join(path, 'clang' + ('.exe' if platform.system() == "Windows" else ''))):
                return path
        
        return None
    
    @staticmethod
    def run_opt(input_file, output_file, passes):
        """Run LLVM opt tool with specified passes"""
        llvm_bin = LLVMTools.find_llvm_bin()
        if not llvm_bin:
            raise FileNotFoundError("LLVM tools not found")
        
        opt_cmd = os.path.join(llvm_bin, 'opt' + ('.exe' if platform.system() == "Windows" else ''))
        
        cmd = [opt_cmd, '-S', input_file]
        for p in passes:
            cmd.append(f'-{p}')
        cmd.extend(['-o', output_file])
        
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result
    
    @staticmethod
    def run_clang(input_file, output_file):
        """Run LLVM clang to compile IR to executable"""
        llvm_bin = LLVMTools.find_llvm_bin()
        if not llvm_bin:
            raise FileNotFoundError("LLVM tools not found")
        
        clang_cmd = os.path.join(llvm_bin, 'clang' + ('.exe' if platform.system() == "Windows" else ''))
        
        cmd = [clang_cmd, input_file, '-o', output_file]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result
    
    @staticmethod
    def measure_runtime(ir_code):
        """Compile and measure runtime of LLVM IR code"""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.ll', delete=False) as ir_file:
            ir_file.write(ir_code)
            ir_file.flush()
            ir_path = ir_file.name
        
        exec_path = ir_path + '.exe' if platform.system() == "Windows" else ir_path + '.out'
        
        # Compile IR to executable
        compile_result = LLVMTools.run_clang(ir_path, exec_path)
        
        if compile_result.returncode != 0:
            os.remove(ir_path)
            try:
                os.remove(exec_path)
            except:
                pass
            return 1e6, compile_result.stderr.decode()
        
        # Run the executable and measure time
        import time
        start = time.time()
        run_result = subprocess.run([exec_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        end = time.time()
        
        # Cleanup
        os.remove(ir_path)
        os.remove(exec_path)
        
        if run_result.returncode != 0:
            return 1e6, run_result.stderr.decode()
        
        return end - start, None
''')
    
    print(f"Created LLVM wrapper at {wrapper_dir}")
    return True

def update_optimizer_env():
    """Update optimizer_env.py to use the LLVM wrapper"""
    optimizer_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rl_optimizer', 'optimizer_env.py')
    
    # Read current file
    with open(optimizer_env_path, 'r') as f:
        content = f.read()
    
    # Add import for LLVM wrapper
    if 'from llvm_wrapper.llvm_tools import LLVMTools' not in content:
        import_line = 'from llvm_wrapper.llvm_tools import LLVMTools'
        import_section_end = content.find('class MLGOEnvironment')
        if import_section_end == -1:
            print("Could not find class definition in optimizer_env.py")
            return False
        
        # Insert import after other imports
        lines = content[:import_section_end].split('\n')
        for i in range(len(lines)-1, -1, -1):
            if lines[i].startswith('import ') or lines[i].startswith('from '):
                lines.insert(i+1, import_line)
                break
        
        new_import_section = '\n'.join(lines)
        content = new_import_section + content[import_section_end:]
    
    # Update _measure_runtime method to use LLVMTools
    measure_runtime_start = content.find('def _measure_runtime')
    if measure_runtime_start == -1:
        print("Could not find _measure_runtime method in optimizer_env.py")
        return False
    
    # Find the end of the method
    next_method_start = content.find('def ', measure_runtime_start + 1)
    if next_method_start == -1:
        next_method_start = len(content)
    
    # Replace the method
    new_method = '''    def _measure_runtime(self, ir_code: str) -> float:
        """
        Measure runtime of compiled IR code using LLVM wrapper
        """
        try:
            runtime, error = LLVMTools.measure_runtime(ir_code)
            if error:
                print(f"Warning: Error during runtime measurement: {error}")
            return runtime
        except Exception as e:
            print(f"Error measuring runtime: {e}")
            # Fall back to static analysis
            return self._estimate_performance_statically(ir_code)
    
    def _estimate_performance_statically(self, ir_code: str) -> float:
        """
        Estimate performance using static code analysis when runtime measurement is not possible
        """
        # Count various operations that affect performance
        instruction_count = len(re.findall(r'\\s+%\\d+\\s+=', ir_code))
        memory_ops = len(re.findall(r'\\balloca\\b|\\bload\\b|\\bstore\\b', ir_code))
        branch_ops = len(re.findall(r'\\bbr\\b|\\bswitch\\b', ir_code))
        call_ops = len(re.findall(r'\\bcall\\b', ir_code))
        loop_markers = len(re.findall(r'\\bloop\\b', ir_code))
        
        # Simple performance model: each operation type has a different weight
        estimated_runtime = (
            0.01 * instruction_count +
            0.05 * memory_ops +
            0.02 * branch_ops +
            0.1 * call_ops +
            0.5 * loop_markers
        )
        
        # Add some randomness to simulate variation in runtime
        noise = np.random.normal(0, 0.05) * estimated_runtime
        estimated_runtime += noise
        
        return max(0.001, estimated_runtime)  # Ensure positive runtime'''
    
    content = content[:measure_runtime_start] + new_method + content[next_method_start:]
    
    # Write updated file
    with open(optimizer_env_path, 'w') as f:
        f.write(content)
    
    print(f"Updated {optimizer_env_path} to use LLVM wrapper")
    return True

def main():
    print("Setting up LLVM for the compiler project...")
    
    # Setup LLVM
    if not setup_project_llvm():
        print("\nWarning: LLVM setup was not successful.")
        print("The project will use static analysis for performance estimation.")
    
    # Create LLVM wrapper
    create_llvm_wrapper()
    
    # Update optimizer_env.py
    update_optimizer_env()
    
    print("\nSetup completed!")
    print("\nTo run your project:")
    print("1. First run 'set_llvm_path.bat' to set the LLVM path")
    print("2. Then run your Python scripts as usual")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
