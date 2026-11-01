# Repository Guidelines

## Design Principles
- Prefer the smallest viable change that satisfies the requirement.
- Avoid modifying code for unrelated robots or devices unless there is no practical alternative.
- Keep device-specific logic and robot-specific logic separated. A teleoperation device should remain reusable across robots.
- Prefer extending or adapting existing implementations over introducing parallel infrastructure.
- Reuse existing assets and integration points before adding new runtime layers, servers, or abstractions.

## Scope Control
- When implementing support for a new robot-device combination, keep shared code unchanged unless the change is broadly useful.
- If behavior differs for only one robot, isolate that difference in the robot-specific layer or example code.
- If behavior differs for only one teleoperation device, isolate that difference in the device-specific layer.

## Integration Strategy
- Follow existing LeRobot patterns for configs, factories, and script wiring.
- Prefer thin adapters around existing components rather than reimplementing mature functionality.
- When an existing module already provides the needed data or lifecycle, build on top of it instead of bypassing it.
- If an existing dependency has rough edges, prefer local adaptation around it before editing the dependency itself.

## Safety and Operability
- Favor behavior that fails clearly over behavior that silently appears to work while doing nothing.
- Keep hardware-safety behavior conservative in shared code.
- If a specific example or workflow needs softer behavior for usability, scope that relaxation locally rather than weakening shared defaults.

## Tuning and Iteration
- Treat teleoperation mapping, scaling, and motion feel as hardware-tuning concerns that should be easy to adjust without redesigning the architecture.
- Keep startup assumptions explicit and configurable.
- Preserve the ability to debug the data path end-to-end: input arrival, enable conditions, mapping, and final robot command.

## Validation
- Use the lightest validation that matches the task and available hardware.
- For hardware integrations, record what was validated in software and what still requires on-device confirmation.

## Commit Style
- Prefer concise Conventional Commit messages.
- When asked to write a commit message, always return one that follows the Conventional Commits format.
- If a feature is usable but still being tuned, make that clear in the commit message rather than overstating completeness.
