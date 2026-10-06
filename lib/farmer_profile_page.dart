import 'package:flutter/material.dart';

class FarmerProfilePage extends StatefulWidget {
  const FarmerProfilePage({super.key});

  @override
  State<FarmerProfilePage> createState() => _FarmerProfilePageState();
}

class _FarmerProfilePageState extends State<FarmerProfilePage> {
  final TextEditingController nameController = TextEditingController();
  final TextEditingController villageController = TextEditingController();
  final TextEditingController cropController = TextEditingController();
  final TextEditingController phoneController = TextEditingController();

  @override
  void dispose() {
    nameController.dispose();
    villageController.dispose();
    cropController.dispose();
    phoneController.dispose();
    super.dispose();
  }

  void saveProfile() {
    if (nameController.text.trim().isEmpty ||
        villageController.text.trim().isEmpty ||
        cropController.text.trim().isEmpty ||
        phoneController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('ದಯವಿಟ್ಟು ಎಲ್ಲಾ ಮಾಹಿತಿಯನ್ನು ನಮೂದಿಸಿ'),
        ),
      );
      return;
    }

    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('ರೈತ ಪ್ರೊಫೈಲ್ ಯಶಸ್ವಿಯಾಗಿ ಉಳಿಸಲಾಗಿದೆ ✅'),
      ),
    );
  }

  Widget buildTextField({
    required TextEditingController controller,
    required String label,
    required String hint,
    required IconData icon,
    TextInputType keyboardType = TextInputType.text,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 18),
      child: TextField(
        controller: controller,
        keyboardType: keyboardType,
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          prefixIcon: Icon(icon),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('ರೈತ ಪ್ರೊಫೈಲ್'),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            const CircleAvatar(
              radius: 45,
              backgroundColor: Colors.green,
              child: Icon(
                Icons.person,
                size: 55,
                color: Colors.white,
              ),
            ),

            const SizedBox(height: 15),

            const Text(
              'ರೈತ ಸ್ನೇಹಿತರೆ, ನಿಮ್ಮ ಮಾಹಿತಿ ನೀಡಿ',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
              textAlign: TextAlign.center,
            ),

            const SizedBox(height: 25),

            buildTextField(
              controller: nameController,
              label: 'ರೈತನ ಹೆಸರು',
              hint: 'ನಿಮ್ಮ ಹೆಸರು ನಮೂದಿಸಿ',
              icon: Icons.person,
            ),

            buildTextField(
              controller: villageController,
              label: 'ಗ್ರಾಮ / ಸ್ಥಳ',
              hint: 'ನಿಮ್ಮ ಗ್ರಾಮ ಅಥವಾ ಸ್ಥಳ',
              icon: Icons.location_on,
            ),

            buildTextField(
              controller: cropController,
              label: 'ಮುಖ್ಯ ಬೆಳೆ',
              hint: 'ಉದಾ: ಅಕ್ಕಿ, ತೆಂಗು, ಅಡಿಕೆ',
              icon: Icons.agriculture,
            ),

            buildTextField(
              controller: phoneController,
              label: 'ಮೊಬೈಲ್ ಸಂಖ್ಯೆ',
              hint: '10 ಅಂಕಿಗಳ ಮೊಬೈಲ್ ಸಂಖ್ಯೆ',
              icon: Icons.phone,
              keyboardType: TextInputType.phone,
            ),

            const SizedBox(height: 10),

            SizedBox(
              width: double.infinity,
              height: 52,
              child: ElevatedButton.icon(
                onPressed: saveProfile,
                icon: const Icon(Icons.save),
                label: const Text(
                  'ಪ್ರೊಫೈಲ್ ಉಳಿಸಿ',
                  style: TextStyle(fontSize: 17),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  foregroundColor: Colors.white,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}