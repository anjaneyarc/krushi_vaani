import 'package:flutter/material.dart';

    class FarmerNotificationsPage extends StatelessWidget {
    const FarmerNotificationsPage({super.key});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'ರೈತ ಮಾಹಿತಿ',
          style: TextStyle(fontWeight: FontWeight.bold),
        ),
        backgroundColor: Colors.green,
        foregroundColor: Colors.white,
      ),

      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _notificationCard(
            icon: Icons.account_balance,
            title: 'ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು',
            description:
                'ರೈತರಿಗೆ ಲಭ್ಯವಿರುವ ಹೊಸ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು ಮತ್ತು ಸಹಾಯಧನದ ಮಾಹಿತಿ.',
            color: Colors.green,
          ),

          _notificationCard(
            icon: Icons.agriculture,
            title: 'ಕೃಷಿ ಮಾಹಿತಿ',
            description:
                'ಬೆಳೆ, ಮಣ್ಣು, ನೀರಾವರಿ ಮತ್ತು ಕೃಷಿ ನಿರ್ವಹಣೆಗೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಮುಖ ಮಾಹಿತಿ.',
            color: Colors.orange,
          ),

          _notificationCard(
            icon: Icons.warning_amber_rounded,
            title: 'ಕೃಷಿ ಎಚ್ಚರಿಕೆ',
            description:
                'ಕೀಟಗಳು, ರೋಗಗಳು ಮತ್ತು ಹವಾಮಾನ ಬದಲಾವಣೆಗಳ ಬಗ್ಗೆ ಪ್ರಮುಖ ಎಚ್ಚರಿಕೆಗಳು.',
            color: Colors.red,
          ),

          _notificationCard(
            icon: Icons.calendar_month,
            title: 'ಪ್ರಮುಖ ದಿನಗಳು',
            description:
                'ಕೃಷಿಗೆ ಸಂಬಂಧಿಸಿದ ಪ್ರಮುಖ ದಿನಗಳು ಮತ್ತು ಕಾರ್ಯಕ್ರಮಗಳ ಮಾಹಿತಿ.',
            color: Colors.blue,
          ),
        ],
      ),
    );
  }

  Widget _notificationCard({
    required IconData icon,
    required String title,
    required String description,
    required Color color,
  }) {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      elevation: 3,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            CircleAvatar(
              radius: 28,
              backgroundColor: color.withOpacity(0.15),
              child: Icon(
                icon,
                color: color,
                size: 30,
              ),
            ),

            const SizedBox(width: 15),

            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: const TextStyle(
                      fontSize: 19,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  const SizedBox(height: 7),

                  Text(
                    description,
                    style: const TextStyle(
                      fontSize: 15,
                      height: 1.4,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}