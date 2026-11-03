"""One-shot ROS2 subscriber for hobot_llamacpp's final PerceptionTargets result.

Run with the system Python after sourcing TogetheROS.Bot. Its stdout protocol is
READY followed by one JSON line; stderr is reserved for diagnostics.
"""
import json
import sys


def main():
    if len(sys.argv) != 2 or not sys.argv[1].startswith('/vlm_agent_x5/'):
        raise SystemExit('需要本次运行专用的ROS话题名')
    import rclpy
    from ai_msgs.msg import PerceptionTargets

    rclpy.init()
    node = rclpy.create_node('vlm_agent_x5_result_listener')
    answer = []

    def received(message):
        for target in message.targets:
            value = target.type.strip()
            if value:
                answer.append(value)
                print(json.dumps({'topic': sys.argv[1], 'answer': value}, ensure_ascii=False), flush=True)
                break

    subscription = node.create_subscription(PerceptionTargets, sys.argv[1], received, 10)
    print('READY', flush=True)
    try:
        while rclpy.ok() and not answer:
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
    return 0 if answer else 2


if __name__ == '__main__':
    sys.exit(main())
