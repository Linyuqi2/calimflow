"""
Test suite for vasculature generation component.

This module tests the Python implementation of blood vessel generation
against MATLAB reference outputs.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.parameters import check_vol_params, check_vasc_params
from .simulate_blood_vessels import VasculatureGenerator


class TestVasculatureGenerator:
    """Test the VasculatureGenerator class"""

    def test_initialization(self):
        """Test basic initialization"""
        vol_params = check_vol_params({'vol_sz': [100, 100, 50], 'vol_depth': 150})
        vasc_params = check_vasc_params({})

        generator = VasculatureGenerator(vol_params, vasc_params)

        assert generator.vres == vol_params['vres']
        assert generator.network is not None

    def test_parameter_initialization(self):
        """Test parameter initialization"""
        vol_params = check_vol_params({'vol_sz': [50, 50, 25], 'vol_depth': 100})
        vasc_params = check_vasc_params({})

        generator = VasculatureGenerator(vol_params, vasc_params)
        generator._initialize_parameters()

        # Check that network parameters are set
        assert generator.network.size.shape == (3,)
        assert generator.network.vol_sz.shape == (3,)
        assert hasattr(generator, 'vp')

    def test_vessel_count_calculation(self):
        """Test vessel count calculation"""
        vol_params = check_vol_params({'vol_sz': [100, 100, 50], 'vol_depth': 150})
        vasc_params = check_vasc_params({})

        generator = VasculatureGenerator(vol_params, vasc_params)
        generator._initialize_parameters()
        generator._calculate_vessel_counts()

        # Check that vessel counts are reasonable
        assert generator.network.nsource >= 0
        assert generator.network.nvert >= 0
        assert generator.network.nsurf >= 0
        assert generator.network.ncapp >= 0

    def test_vessel_statistics(self):
        """Test vessel statistics generation"""
        vol_params = check_vol_params({'vol_sz': [50, 50, 25], 'vol_depth': 100})
        vasc_params = check_vasc_params({})

        generator = VasculatureGenerator(vol_params, vasc_params)
        stats = generator.get_vessel_statistics()

        assert isinstance(stats, dict)
        assert 'total_nodes' in stats
        assert 'total_connections' in stats
        assert 'volume_size' in stats
        assert 'vessel_counts' in stats


class TestVasculatureIntegration:
    """Integration tests for complete vasculature generation"""

    def test_basic_generation(self):
        """Test basic vasculature generation with small volume"""
        vol_params = check_vol_params({
            'vol_sz': [30, 30, 15],  # Very small volume for fast testing
            'vol_depth': 50,
            'random_seed': 42,  # For reproducible results
        })
        vasc_params = check_vasc_params({
            'flag': True,  # Enable vasculature generation
        })

        # This should not raise an exception
        generator = VasculatureGenerator(vol_params, vasc_params)

        # Get basic statistics (should work even with placeholder implementation)
        stats = generator.get_vessel_statistics()
        assert stats['volume_size'].shape == (3,)

    def test_volume_shapes(self):
        """Test that generated volumes have correct shapes"""
        vol_params = check_vol_params({
            'vol_sz': [40, 40, 20],
            'vol_depth': 60,
        })
        vasc_params = check_vasc_params({})

        generator = VasculatureGenerator(vol_params, vasc_params)
        generator._initialize_parameters()

        # Check volume size calculations
        expected_size = np.array([40, 40, 80]) * vol_params['vres']  # vol_depth + vol_sz[2]
        np.testing.assert_array_equal(generator.network.size, expected_size)


class TestMATLABCompatibility:
    """
    Test compatibility with MATLAB reference outputs.

    These tests require MATLAB reference files to be generated first.
    """

    @pytest.fixture
    def matlab_ref_dir(self):
        """Get the directory containing MATLAB reference outputs"""
        ref_dir = Path(__file__).parent.parent.parent / 'tests' / 'matlab_references'
        return ref_dir

    def load_matlab_reference(self, ref_dir: Path, test_name: str):
        """Load MATLAB reference output from JSON file"""
        ref_file = ref_dir / f"{test_name}.json"
        if not ref_file.exists():
            pytest.skip(f"MATLAB reference file not found: {ref_file}")

        import json
        with open(ref_file, 'r') as f:
            return json.load(f)

    @pytest.mark.matlab
    def test_vasculature_basic_compatibility(self, matlab_ref_dir):
        """Test basic vasculature generation compatibility"""
        # For now, just check that we can run without errors
        # Full compatibility testing will be added once the implementation is complete

        vol_params = check_vol_params({
            'vol_sz': [30, 30, 15],
            'vol_depth': 50,
            'random_seed': 42,
        })
        vasc_params = check_vasc_params({
            'flag': True,
        })

        generator = VasculatureGenerator(vol_params, vasc_params)

        # Should not raise an exception
        assert generator is not None

        # TODO: Add full compatibility testing once implementation is complete
        pytest.skip("Full implementation needed for compatibility testing")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
