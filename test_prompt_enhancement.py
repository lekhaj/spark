#!/usr/bin/env python3
"""
Test script for 3D-optimized prompt enhancement
"""

def test_prompt_enhancement():
    """Test the prompt enhancement function with various inputs"""
    
    # Test data - various prompts that users might enter
    test_prompts = [
        "a red sports car",
        "wooden dining chair with leather cushion",
        "ceramic vase with blue patterns",
        "a house in a landscape with trees",  # Should remove landscape elements
        "modern lamp on a table in a room",   # Should remove room/environment
        "golden crown with gems, black background",  # Should remove black background
        "vintage camera, detailed environment",  # Should remove environment
        "simple coffee mug",
        "futuristic robot standing outdoor",  # Should remove outdoor
        ""  # Empty prompt test
    ]
    
    print("🧪 Testing 3D-Optimized Prompt Enhancement")
    print("=" * 80)
    
    # Import the function (assuming this script is in the same directory)
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from merged_gradio_app import enhance_prompt_for_3d_generation
        
        for i, original in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}:")
            print(f"   Original: '{original}'")
            
            enhanced = enhance_prompt_for_3d_generation(original)
            print(f"   Enhanced: '{enhanced}'")
            
            # Check if 3D keywords were added
            if "3d render" in enhanced.lower() and "photorealistic" in enhanced.lower():
                print("   ✅ 3D keywords added successfully")
            else:
                print("   ❌ 3D keywords missing")
            
            # Check if white background was added
            if "white background" in enhanced.lower():
                print("   ✅ White background specified")
            else:
                print("   ❌ White background missing")
            
            # Check if conflicting terms were removed
            conflicting_found = any(term in enhanced.lower() for term in 
                                  ["landscape", "indoor", "outdoor", "room", "environment", "black background"])
            if not conflicting_found:
                print("   ✅ Conflicting terms removed")
            else:
                print("   ⚠️ Some conflicting terms may remain")
        
        print(f"\n🎉 Prompt enhancement testing completed!")
        
    except ImportError as e:
        print(f"❌ Could not import enhancement function: {e}")
        print("Make sure you're running this from the project root directory")
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    test_prompt_enhancement()
