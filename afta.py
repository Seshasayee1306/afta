import numpy as np
import tensorflow as tf

# Example input dimension, replace with your actual data shape
input_dim = 10  # e.g., 10 features in your data

# Initialize global model
global_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


def local_training(data, labels, epochs=5, batch_size=32):
    local_model = tf.keras.models.clone_model(global_model)
    local_model.set_weights(global_model.get_weights())

    local_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    local_model.fit(data, labels, epochs=epochs, batch_size=batch_size)

    return local_model.get_weights()


def aggregate_updates(updates):
    new_weights = [np.mean([u[i] for u in updates], axis=0) for i in range(len(updates[0]))]
    global_model.set_weights(new_weights)


def fine_tune_model(data, labels, epochs=3):
    global_model.fit(data, labels, epochs=epochs)


def evaluate_model(data, labels):
    loss, accuracy = global_model.evaluate(data, labels)
    print(f'Evaluation - Loss: {loss}, Accuracy: {accuracy}')
    return accuracy


# Example data with names and stress reasons
people = [
    {'name': 'staff1', 'data': np.random.rand(10, input_dim), 'labels': np.random.randint(2, size=10),
     'stress_reason': 'Workload'},
    {'name': 'staff2', 'data': np.random.rand(10, input_dim), 'labels': np.random.randint(2, size=10),
     'stress_reason': 'Deadlines'},
    {'name': 'staff3', 'data': np.random.rand(10, input_dim), 'labels': np.random.randint(2, size=10),
     'stress_reason': 'Work-life balance'},
    {'name': 'staff4', 'data': np.random.rand(10, input_dim), 'labels': np.random.randint(2, size=10),
     'stress_reason': 'Management'},
]

# Split data into training and test sets
train_data = np.concatenate([person['data'] for person in people])
train_labels = np.concatenate([person['labels'] for person in people])
test_data = np.random.rand(20, input_dim)
test_labels = np.random.randint(2, size=20)

# Federated Learning Process
num_rounds = 5
for round in range(num_rounds):
    local_updates = []
    for person in people:
        print(f"Training for {person['name']} due to {person['stress_reason']}")
        local_update = local_training(person['data'], person['labels'])
        local_updates.append(local_update)
    aggregate_updates(local_updates)

    # Optional: Fine-tune on specific person's data if needed
    fine_tune_model(people[0]['data'], people[0]['labels'])

    # Evaluation
    accuracy = evaluate_model(test_data, test_labels)
    if accuracy > 0.95:  # Example condition to stop early
        break

print("Final global model is ready for deployment.")
