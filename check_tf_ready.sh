#!/bin/bash
# Script to check if all required TF transforms are available before starting Nav2

echo "Checking TF transforms..."

# Check map->odom (from Cartographer)
echo -n "Checking map->odom transform... "
OUTPUT=$(timeout 5 ros2 run tf2_ros tf2_echo map odom 2>&1 | grep -E "Translation:|At time" | head -2)
if echo "$OUTPUT" | grep -q "Translation:"; then
    echo "✓ OK"
else
    echo "✗ NOT READY (transform not found)"
    exit 1
fi

# Check odom->base_link (through base_footprint)
echo -n "Checking odom->base_link transform... "
OUTPUT=$(timeout 5 ros2 run tf2_ros tf2_echo odom base_link 2>&1 | grep -E "Translation:|At time" | head -2)
if echo "$OUTPUT" | grep -q "Translation:"; then
    echo "✓ OK"
else
    echo "✗ NOT READY (transform not found)"
    exit 1
fi

# Check if map topic exists and is being published
echo -n "Checking /map topic... "
if timeout 3 ros2 topic echo /map --once > /dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ NOT READY (topic not publishing)"
    exit 1
fi

echo ""
echo "All transforms ready! Safe to start Nav2."
exit 0

