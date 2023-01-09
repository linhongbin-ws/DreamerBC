- Dreamer + origin: 
  ```sh
  python examples/suture_json.py --baseline Dreamer --preprocess-type origin --clutch -1 --section 1
  ```
- Dreamer + our preprocess + clutch
  ```sh
  python examples/suture_json.py --baseline Dreamer   --section 1
  ```

- DreamerBC + origin
  ```sh
  python examples/suture_json.py --preprocess-type origin --clutch -1 --section 1
  ```

- DreamerBC + our preprocess + clutch
  ```sh
  python examples/suture_json.py --section 1
  ```