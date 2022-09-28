from src.switch_2afc_task import Switch2AFCTask


def test_env(env, block_params):
    for ep in range(sum(block[2] for block in block_params) + 1):
        env.step()
        if env.done:
            break
    return env


if __name__ == "__main__":
    test_blocks = [('LOW', 1.0, 100), ('HIGH', 1.0, 100), ('LOW', 1.0, 100), ('HIGH', 1.0, 100)]
    test_task = Switch2AFCTask(test_blocks)
    test_env(test_task, test_blocks)
    assert (test_task.done is True)
    print("Test passed!")
