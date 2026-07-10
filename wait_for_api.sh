for i in {1..10}; do
  if ss -lptn | grep -q 3001; then
    echo "API is up"
    exit 0
  fi
  sleep 1
done
echo "API is down"
