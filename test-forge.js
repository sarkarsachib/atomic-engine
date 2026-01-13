// Simple test for Atomic Forge module
const { forge } = require('./src/modules/forge/index.js');

/**
 * Runs a smoke test of the Forge generator using a predefined project specification.
 *
 * Builds a sample spec, invokes forge.generate(spec), logs a summary and a preview of generated files and build instructions, and returns the generator output.
 *
 * @returns {Object} The generation result with properties like `summary`, `files` (array of generated files), and `buildInstructions`.
 */
async function testForge() {
  try {
    console.log('🧪 Testing Atomic Forge...');
    
    const spec = {
      name: 'test-api',
      language: 'typescript',
      framework: 'express',
      architecture: 'mvc',
      database: 'postgresql',
      authentication: 'jwt',
      testing: true,
      deployment: ['docker', 'kubernetes'],
      features: ['user-management', 'authentication'],
      modules: ['users', 'auth']
    };

    console.log('📋 Generating project with spec:', spec);
    
    const result = await forge.generate(spec);
    
    console.log('✅ Generation completed successfully!');
    console.log('📊 Summary:', result.summary);
    console.log('📁 Generated files:', result.files.length);
    console.log('📄 Build instructions preview:', result.buildInstructions.substring(0, 200) + '...');
    
    // Show some generated files
    console.log('\n📝 Sample generated files:');
    result.files.slice(0, 5).forEach(file => {
      console.log(`  - ${file.path} (${file.language})`);
    });
    
    return result;
  } catch (error) {
    console.error('❌ Forge test failed:', error);
    throw error;
  }
}

// Run the test
testForge()
  .then(result => {
    console.log('\n🎉 All tests passed!');
    process.exit(0);
  })
  .catch(error => {
    console.error('\n💥 Test failed:', error);
    process.exit(1);
  });