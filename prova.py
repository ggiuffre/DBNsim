from RBM import RBM

def main():
    net = RBM(2, 5)
    examples = [ # XOR trainset
        [1.0,  0.1 ],
        [0.9,  0.0 ],
        [0.0,  1.0 ],
        [0.1,  1.0 ],
        [0.14, 1.1 ],
        [0.82, 0.0 ],
        [1.0,  0.0 ],
        [0.87, 0.0 ],
        [0.01, 1.2 ],
        [0.1,  0.95],
        [0.0,  1.01],
        [1.0,  0.06],
        [1.28, 0.2 ],
        [0.0,  0.9 ]
    ]
    validation = [
        [0.0, 0.0],
        [0.1, 0.2],
        [0.3, 0.1],
        [0.1, 0.9],
        [1.0, 0.0],
        [0.3, 0.8]
    ]
    net.learn(examples)
    print('net:\n', net)
    for test in validation:
        print(test, '-->', net.evaluate(test))



if __name__ == '__main__':
    main()
