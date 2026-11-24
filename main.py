from step_01_dataset import create_synthetic_people_dataset


def main():
    
    print("\n========== STEP 01 DATASET ==========")
    df = create_synthetic_people_dataset()
    print(df.head(10))
    

if __name__ == "__main__":
    main()